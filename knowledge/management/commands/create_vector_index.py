"""Create the Atlas Vector Search index used by VECTOR_BACKEND=mongo."""
import time

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connection
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from ai import gemini_client
from knowledge.models import Chunk


def index_definition(dims):
    return {'fields': [
        {'type': 'vector', 'path': 'embedding', 'numDimensions': dims, 'similarity': 'cosine'},
        {'type': 'filter', 'path': 'organization_id'},
    ]}


class Command(BaseCommand):
    help = 'Create (or --recreate) the $vectorSearch index on the chunk collection (Atlas / mongot only).'

    def add_arguments(self, parser):
        parser.add_argument('--dims', type=int, help='Override vector dimensions (default: probe GEMINI_EMBED_MODEL).')
        parser.add_argument('--recreate', action='store_true', help='Drop an existing index with the same name first.')
        parser.add_argument('--no-wait', action='store_true', help='Return without waiting for the index to be queryable.')

    def handle(self, *args, **opts):
        dims = opts['dims'] or self._probe_dims()
        stored = Chunk.objects.values_list('embedding', flat=True).first()
        if stored and len(stored) != dims:
            raise CommandError(f'Stored chunks have {len(stored)} dims but index would use {dims}; reprocess documents.')

        name = settings.MONGO_VECTOR_INDEX
        db = connection.get_database()
        table = Chunk._meta.db_table
        if table not in db.list_collection_names():
            db.create_collection(table)
        coll = db[table]
        try:
            existing = {ix['name'] for ix in coll.list_search_indexes()}
            if name in existing:
                if not opts['recreate']:
                    self.stdout.write(f'Index {name!r} already exists (use --recreate to rebuild).')
                    return
                coll.drop_search_index(name)
                self._wait(coll, name, gone=True)
            coll.create_search_index(SearchIndexModel(definition=index_definition(dims), name=name, type='vectorSearch'))
        except OperationFailure as exc:
            raise CommandError(f'Search indexes need Atlas or MongoDB with mongot: {exc}') from exc

        self.stdout.write(f'Creating {name!r} on {table}.embedding ({dims} dims, cosine, filter organization_id)...')
        if not opts['no_wait']:
            self._wait(coll, name)
        self.stdout.write(self.style.SUCCESS(f'Index {name!r} ready. Set VECTOR_BACKEND=mongo.'))

    def _probe_dims(self):
        try:
            dims = len(gemini_client.embed_query('dimension probe'))
        except gemini_client.GeminiError as exc:
            raise CommandError(f'Could not probe {settings.GEMINI_EMBED_MODEL} ({exc}); pass --dims.') from exc
        self.stdout.write(f'{settings.GEMINI_EMBED_MODEL} returns {dims}-dim vectors.')
        return dims

    def _wait(self, coll, name, gone=False, timeout=300):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            found = [ix for ix in coll.list_search_indexes(name)]
            if gone and not found:
                return
            if not gone and found and found[0].get('queryable'):
                return
            time.sleep(5)
        raise CommandError(f'Timed out waiting for index {name!r}.')
