$(function () {
  var $chat = $('#chat');
  var apiUrl = $chat.data('api-url');
  var token = $chat.data('token');
  var $messages = $('#chat-messages');
  var $input = $('#chat-input');
  var $send = $('#chat-send');

  var OUTCOMES = {
    answered: ['Answered', 'text-bg-success'],
    no_answer: ['No answer found', 'text-bg-secondary'],
    needs_review: ['Needs review', 'text-bg-warning'],
    error: ['Error', 'text-bg-danger'],
  };

  function escapeHtml(s) {
    return $('<div>').text(s).html();
  }

  // Inline Markdown on already-escaped text: **bold**, *italic*, `code`, [doc p.N] citations.
  function inline(s) {
    return escapeHtml(s)
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, '$1<em>$2</em>')
      .replace(/\[([^\[\]]+?\s+p\.\s*\d+)\]/g, '<span class="cite">$1</span>');
  }

  function splitRow(line) {
    return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(function (c) { return c.trim(); });
  }

  // Safe Markdown subset → HTML. Every text fragment goes through escapeHtml; no raw HTML passes.
  function renderText(text) {
    var lines = String(text).split(/\r?\n/), out = [], i = 0;
    var listRe = /^(\s*)([-*•]|\d+[.)])\s+(.*)$/;
    while (i < lines.length) {
      var line = lines[i];
      if (!line.trim()) { i++; continue; }
      var h = line.match(/^\s*(#{1,6})\s+(.*)$/);
      if (h) { out.push('<div class="md-h">' + inline(h[2]) + '</div>'); i++; continue; }
      // Table: header row followed by a |---|---| separator.
      if (line.indexOf('|') !== -1 && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
        var head = splitRow(line), rows = [];
        i += 2;
        while (i < lines.length && lines[i].indexOf('|') !== -1 && lines[i].trim()) { rows.push(splitRow(lines[i])); i++; }
        var t = '<div class="table-responsive"><table class="table table-sm table-bordered table-striped align-middle mb-2"><thead><tr>';
        head.forEach(function (c) { t += '<th>' + inline(c) + '</th>'; });
        t += '</tr></thead><tbody>';
        rows.forEach(function (r) {
          t += '<tr>';
          head.forEach(function (_, k) { t += '<td>' + inline(r[k] || '') + '</td>'; });
          t += '</tr>';
        });
        out.push(t + '</tbody></table></div>');
        continue;
      }
      if (listRe.test(line)) {
        // Nested lists by indentation; a stack of open list tags.
        var stack = [], html = '';
        while (i < lines.length && listRe.test(lines[i])) {
          var m = lines[i].match(listRe), depth = Math.floor(m[1].replace(/\t/g, '  ').length / 2);
          var tag = /\d/.test(m[2]) ? 'ol' : 'ul';
          while (stack.length > depth + 1) html += '</li></' + stack.pop() + '>';
          if (stack.length === depth + 1 && stack[depth] !== tag) { html += '</li></' + stack.pop() + '>'; }
          if (stack.length < depth + 1) {
            while (stack.length < depth + 1) { html += '<' + tag + '>'; stack.push(tag); }
          } else { html += '</li>'; }
          html += '<li>' + inline(m[3]);
          i++;
        }
        while (stack.length) html += '</li></' + stack.pop() + '>';
        out.push(html.replace(/<(ul|ol)><\/li>/g, '<$1>'));
        continue;
      }
      var para = [];
      while (i < lines.length && lines[i].trim() && !listRe.test(lines[i]) && !/^\s*#{1,6}\s/.test(lines[i]) &&
             !(lines[i].indexOf('|') !== -1 && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1]))) {
        para.push(inline(lines[i])); i++;
      }
      out.push('<p>' + para.join('<br>') + '</p>');
    }
    return out.join('');
  }

  function scrollDown() { $messages.scrollTop($messages[0].scrollHeight); }

  function addMessage(role, $content) {
    var $msg = $('<div class="msg">').addClass(role === 'user' ? 'msg-user' : 'msg-bot');
    $msg.append($('<div class="bubble">').append($content));
    $messages.append($msg);
    scrollDown();
    return $msg;
  }

  function botAnswer(data) {
    var $c = $('<div>');
    var o = OUTCOMES[data.outcome] || OUTCOMES.error;
    $c.append($('<span class="badge mb-1">').addClass(o[1]).text(o[0]));
    $c.append($('<div>').html(renderText(data.answer || '')));
    if (data.references && data.references.length) {
      var $ul = $('<ul class="mb-0">');
      data.references.forEach(function (r) {
        $ul.append($('<li>').text(r.document + ' — Page ' + r.page));
      });
      $c.append($('<div class="sources">').append('<div class="fw-semibold">Sources</div>').append($ul));
    }
    if (data.question_type === 'decision' && data.decisions && data.decisions.length) {
      $c.append(decisionDetails(data));
    }
    addMessage('bot', $c);
  }

  function formatValue(d) {
    if (d.error) return 'could not be evaluated';
    if (d.value === true) return 'met';
    if (d.value === false) return 'not met';
    return String(d.value);
  }

  // Collapsible list: id, kind, value, confidence (+ missing facts) — all text-escaped via .text().
  function decisionDetails(data) {
    var $d = $('<details class="decision-details">').append($('<summary>').text('Decision details'));
    var $ul = $('<ul class="mb-0">');
    data.decisions.forEach(function (d) {
      var conf = typeof d.confidence === 'number' ? Math.round(d.confidence * 100) + '%' : '—';
      var $li = $('<li>').append($('<code>').text(d.id)).append(document.createTextNode(
        ' · ' + d.kind + ' · ' + formatValue(d) + ' · confidence ' + conf));
      if (d.evidence_ref) $li.append($('<div class="text-muted">').text(d.evidence_ref));
      $ul.append($li);
    });
    $d.append($ul);
    if (data.missing_facts && data.missing_facts.length) {
      $d.append($('<div class="mt-1">').text('Missing information: ' + data.missing_facts.join(', ')));
    }
    return $d;
  }

  $('#chat-form').on('submit', function (e) {
    e.preventDefault();
    var q = $.trim($input.val());
    if (!q) return;
    addMessage('user', $('<div>').text(q));
    $input.val('');
    $send.prop('disabled', true);
    var $typing = addMessage('bot', $('<div class="typing" aria-label="Assistant is typing"><span></span><span></span><span></span></div>'));
    $.ajax({
      url: apiUrl,
      method: 'POST',
      contentType: 'application/json',
      dataType: 'json',
      data: JSON.stringify({ token: token, question: q }),
      timeout: 60000,
    }).done(botAnswer).fail(function (xhr) {
      var msg = (xhr.responseJSON && xhr.responseJSON.error) || 'Something went wrong. Please try again.';
      botAnswer({ outcome: 'error', answer: msg, references: [] });
    }).always(function () {
      $typing.remove();
      $send.prop('disabled', false);
      $input.trigger('focus');
    });
  });
});
