$(function () {
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    var $tmp = $('<textarea>').val(text).css({ position: 'fixed', opacity: 0 }).appendTo('body');
    $tmp[0].select();
    var ok = document.execCommand('copy');
    $tmp.remove();
    return ok ? $.Deferred().resolve().promise() : $.Deferred().reject().promise();
  }

  $('.js-copy').on('click', function () {
    var $btn = $(this);
    var $target = $($btn.data('target'));
    var text = $target.is('input') ? $target.val() : $target.text();
    $.when(copyText(text)).then(function () {
      $btn.text('Copied!');
      setTimeout(function () { $btn.text('Copy'); }, 1500);
    }, function () {
      $btn.text('Copy failed');
    });
  });

  var $confirm = $('#regenerate-confirm');
  var $error = $('#regenerate-error');
  $('#regenerate-modal').on('show.bs.modal', function () { $error.addClass('d-none'); });

  $confirm.on('click', function () {
    $confirm.prop('disabled', true);
    $.post($confirm.data('url')).done(function (data) {
      $('#assistant-url').val(data.url);
      $('#iframe-snippet').text(data.iframe_snippet);
      $('#script-snippet').text(data.script_snippet);
      $('#preview-link').attr('href', data.url).removeClass('d-none');
      $('.js-copy').prop('disabled', false);
      $('#token-notice').removeClass('alert-info').addClass('alert-success')
        .html('New link generated; the old one no longer works. <strong>Copy it now</strong> — for security it is shown only once.');
      bootstrap.Modal.getInstance(document.getElementById('regenerate-modal')).hide();
    }).fail(function (xhr) {
      var msg = (xhr.responseJSON && xhr.responseJSON.error) || 'Could not generate a new link. Please try again.';
      $error.text(msg).removeClass('d-none');
    }).always(function () {
      $confirm.prop('disabled', false);
    });
  });
});
