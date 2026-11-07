// Attach Django's CSRF token to every same-origin unsafe AJAX request.
$.ajaxSetup({
  beforeSend: function (xhr, settings) {
    if (!/^(GET|HEAD|OPTIONS|TRACE)$/i.test(settings.type) && !this.crossDomain) {
      xhr.setRequestHeader('X-CSRFToken', $('meta[name="csrf-token"]').attr('content'));
    }
  },
});
