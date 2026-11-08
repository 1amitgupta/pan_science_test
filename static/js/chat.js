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

  // Escape everything, then allow only line breaks and simple "- " / "* " bullet lists.
  function renderText(text) {
    var html = [], list = [];
    function flush() {
      if (list.length) { html.push('<ul>' + list.join('') + '</ul>'); list = []; }
    }
    String(text).split(/\r?\n/).forEach(function (line) {
      var m = line.match(/^\s*[-*•]\s+(.*)$/);
      if (m) { list.push('<li>' + escapeHtml(m[1]) + '</li>'); return; }
      flush();
      html.push(escapeHtml(line) + '<br>');
    });
    flush();
    return html.join('').replace(/(<br>)+$/, '');
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
