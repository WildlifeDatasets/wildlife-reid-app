/* One shared size and bbox mode for every preview, including hidden carousel slides. */
(() => {
  'use strict';
  const controls = document.querySelector('[data-bbox-controls]');
  if (!controls) return;
  const smaller = controls.querySelector('.js-preview-smaller');
  const larger = controls.querySelector('.js-preview-larger');
  const toggle = controls.querySelector('.js-bbox-zoom');
  let previewScale = 1, zoomed = false;
  const previews = Array.from(document.querySelectorAll('[data-bbox-widget]')).map(widget => {
    const frame = widget.querySelector('[data-bbox]');
    const image = frame?.querySelector('img');
    if (!frame || !image) return null;
    const box = frame.dataset.bbox.split(',').map(Number);
    let hasBox = box.length === 4 && box.every(Number.isFinite) && box[2] > 0 && box[3] > 0;
    const thumbnail = image.getAttribute('src');
    const imageStyle = image.getAttribute('style');
    const frameStyle = frame.getAttribute('style');
    const wrapper = widget.classList.contains('similar-pair-thumbnail-widget') ? widget : null;
    const wrapperStyle = wrapper?.getAttribute('style');
    let baseHeight, baseWidth;
    function restoreStyle(element, value) {
      if (value == null) element.removeAttribute('style');
      else element.setAttribute('style', value);
    }
    function position() {
      if (!zoomed || !hasBox || !image.naturalWidth || !image.naturalHeight || !frame.clientWidth || !frame.clientHeight) return;
      const scale = Math.min(frame.clientWidth / (image.naturalWidth * box[2] * 1.1), frame.clientHeight / (image.naturalHeight * box[3] * 1.1));
      const x = frame.clientWidth / 2 - image.naturalWidth * box[0] * scale;
      const y = frame.clientHeight / 2 - image.naturalHeight * box[1] * scale;
      Object.assign(image.style, {position: 'absolute', maxWidth: 'none',
        transform: 'translate(' + x + 'px,' + y + 'px) scale(' + scale + ')'});
      image.style.setProperty('width', image.naturalWidth + 'px', 'important');
      image.style.setProperty('height', image.naturalHeight + 'px', 'important');
    }
    function resize() {
      if (!baseHeight) {
        const computed = getComputedStyle(frame);
        baseHeight = parseFloat(computed.height) || frame.clientHeight;
        baseWidth = parseFloat(computed.width) || frame.clientWidth;
      }
      if (previewScale === 1) {
        restoreStyle(frame, frameStyle);
        if (wrapper) restoreStyle(wrapper, wrapperStyle);
        baseHeight = baseWidth = null;
      } else if (baseHeight) {
        frame.style.height = baseHeight * previewScale + 'px';
        if (wrapper && baseWidth) {
          wrapper.style.width = wrapper.style.flexBasis = baseWidth * previewScale + 'px';
          frame.style.width = '100%';
        }
      }
      position();
    }
    function applyMode() {
      restoreStyle(image, imageStyle);
      const url = zoomed && hasBox ? frame.dataset.fullSrc || thumbnail : thumbnail;
      if (image.getAttribute('src') !== url) image.setAttribute('src', url);
      position();
    }
    image.addEventListener('load', position);
    image.addEventListener('error', () => {
      if (!zoomed || !hasBox) return;
      hasBox = false;
      applyMode();
      updateControls();
    });
    if (window.ResizeObserver) new ResizeObserver(position).observe(frame);
    else window.addEventListener('resize', position);
    return {resize, applyMode, hasBox: () => hasBox};
  }).filter(Boolean);

  function updateControls() {
    smaller.disabled = !previews.length || previewScale <= .5;
    larger.disabled = !previews.length || previewScale >= 3;
    toggle.disabled = !zoomed && !previews.some(preview => preview.hasBox());
    toggle.setAttribute('aria-pressed', String(zoomed));
    toggle.textContent = zoomed ? 'Full media' : 'Zoom to bbox';
    toggle.classList.toggle('active', zoomed);
  }
  function resizeAll(factor) {
    previewScale = Math.min(3, Math.max(.5, previewScale * factor));
    if (Math.abs(previewScale - 1) < 1e-9) previewScale = 1;
    previews.forEach(preview => preview.resize());
    updateControls();
  }
  smaller.addEventListener('click', () => resizeAll(.8));
  larger.addEventListener('click', () => resizeAll(1.25));
  toggle.addEventListener('click', () => {
    zoomed = !zoomed;
    previews.forEach(preview => preview.applyMode());
    updateControls();
  });
  updateControls();
})();
