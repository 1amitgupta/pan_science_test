(function () {
  if (window.__kaAssistantLoaded) return;
  window.__kaAssistantLoaded = true;
  var chatUrl = "{{ chat_url|escapejs }}";
  var frame = null;
  var open = false;

  var btn = document.createElement('button');
  btn.type = 'button';
  btn.setAttribute('aria-label', 'Open assistant');
  btn.textContent = '💬';
  btn.style.cssText = 'position:fixed;right:20px;bottom:20px;z-index:2147483646;width:56px;height:56px;' +
    'border-radius:50%;border:0;background:#0d6efd;color:#fff;font-size:24px;cursor:pointer;' +
    'box-shadow:0 4px 12px rgba(0,0,0,.25)';

  function toggle() {
    if (!frame) {
      frame = document.createElement('iframe');
      frame.src = chatUrl;
      frame.title = 'Assistant';
      frame.style.cssText = 'position:fixed;right:20px;bottom:88px;z-index:2147483647;' +
        'width:min(380px,calc(100vw - 40px));height:min(560px,calc(100vh - 120px));border:0;' +
        'border-radius:12px;background:#fff;box-shadow:0 8px 24px rgba(0,0,0,.25)';
      document.body.appendChild(frame);
    }
    open = !open;
    frame.style.display = open ? 'block' : 'none';
    btn.textContent = open ? '✕' : '💬';
    btn.setAttribute('aria-label', open ? 'Close assistant' : 'Open assistant');
  }

  btn.addEventListener('click', toggle);
  function mount() { document.body.appendChild(btn); }
  if (document.body) mount(); else document.addEventListener('DOMContentLoaded', mount);
})();
