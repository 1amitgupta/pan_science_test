// Knowledge Base page: AJAX upload with progress, delete with confirm modal.
$(function () {
  var STATUS_CLASS = {
    pending: 'text-bg-secondary', processing: 'text-bg-info',
    ready: 'text-bg-success', failed: 'text-bg-danger',
  };
  var $rows = $('#doc-rows');
  var $alerts = $('#upload-alerts');
  var $progress = $('#upload-progress');
  var $bar = $progress.find('.progress-bar');
  var modal = new bootstrap.Modal('#delete-modal');
  var pendingDelete = null;

  function styleBadges($scope) {
    $scope.find('.badge[class*="status-"]').each(function () {
      var m = this.className.match(/status-(\w+)/);
      if (m) $(this).addClass(STATUS_CLASS[m[1]] || 'text-bg-secondary');
    });
  }

  function setCount(count) {
    $('#doc-count').text(count);
    var full = count >= parseInt($('#doc-limit').data('max'), 10);
    $('#doc-limit').toggleClass('text-bg-secondary', !full).toggleClass('text-bg-warning', full);
    $('#empty-row').toggleClass('d-none', count > 0);
  }

  function showAlert(kind, message) {
    var $a = $('<div class="alert alert-dismissible fade show" role="alert">')
      .addClass('alert-' + kind).text(message)
      .append('<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>');
    $alerts.empty().append($a);
  }

  function errorMessage(xhr, fallback) {
    return (xhr.responseJSON && xhr.responseJSON.error) || fallback;
  }

  function buildRow(doc) {
    var $status = $('<td>').append(
      $('<span class="badge">').addClass('status-' + doc.status).text(doc.status_label));
    if (doc.error) $status.append($('<div class="small text-danger">').text(doc.error));
    var url = window.KB_DELETE_URL.replace('000000000000000000000000', doc.id);
    return $('<tr>').attr('data-id', doc.id).append(
      $('<td class="doc-name">').text(doc.name),
      $('<td>').text(doc.pages),
      $('<td>').text(doc.size),
      $status,
      $('<td>').text(doc.uploaded),
      $('<td class="text-end">').append(
        $('<button type="button" class="btn btn-sm btn-outline-danger js-delete">Delete</button>')
          .attr('data-url', url))
    );
  }

  styleBadges($rows);

  $('#upload-form').on('submit', function (e) {
    e.preventDefault();
    var input = $('#upload-file')[0];
    if (!input.files.length) return showAlert('warning', 'Choose a PDF file first.');
    var data = new FormData(this);
    var $btn = $('#upload-btn').prop('disabled', true);
    $alerts.empty();
    $bar.css('width', '0%').text('0%');
    $progress.removeClass('d-none');

    $.ajax({
      url: this.action, type: 'POST', data: data,
      processData: false, contentType: false, dataType: 'json',
      xhr: function () {
        var xhr = new window.XMLHttpRequest();
        xhr.upload.addEventListener('progress', function (ev) {
          if (!ev.lengthComputable) return;
          var pct = Math.round((ev.loaded / ev.total) * 100);
          $bar.css('width', pct + '%').text(pct < 100 ? pct + '%' : 'Processing…');
        });
        return xhr;
      },
    }).done(function (resp) {
      var $row = buildRow(resp.document);
      styleBadges($row);
      $rows.prepend($row);
      setCount(resp.count);
      input.value = '';
      if (resp.document.status === 'failed') {
        showAlert('warning', 'Uploaded "' + resp.document.name + '" but processing failed: ' + resp.document.error);
      } else {
        showAlert('success', 'Uploaded "' + resp.document.name + '".');
      }
    }).fail(function (xhr) {
      showAlert('danger', errorMessage(xhr, 'Upload failed. Please try again.'));
    }).always(function () {
      $btn.prop('disabled', false);
      $progress.addClass('d-none');
    });
  });

  $rows.on('click', '.js-delete', function () {
    var $row = $(this).closest('tr');
    pendingDelete = { url: $(this).data('url'), $row: $row };
    $('#delete-doc-name').text($row.find('.doc-name').text());
    $('#delete-error').addClass('d-none');
    modal.show();
  });

  $('#delete-confirm').on('click', function () {
    if (!pendingDelete) return;
    var $btn = $(this).prop('disabled', true);
    $.ajax({ url: pendingDelete.url, type: 'POST', dataType: 'json' })
      .done(function (resp) {
        pendingDelete.$row.remove();
        setCount(resp.count);
        modal.hide();
        pendingDelete = null;
      })
      .fail(function (xhr) {
        var msg = xhr.status === 404 ? 'Document not found (it may already be deleted).'
          : errorMessage(xhr, 'Delete failed. Please try again.');
        $('#delete-error').text(msg).removeClass('d-none');
      })
      .always(function () { $btn.prop('disabled', false); });
  });

  setCount(parseInt($('#doc-count').text(), 10));
});
