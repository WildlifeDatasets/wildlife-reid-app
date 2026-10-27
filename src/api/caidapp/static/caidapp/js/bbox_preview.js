/* Read-only zoom of image previews to a prevalidated bounding-box union. */
(() => {
  'use strict';
  document.querySelectorAll('[data-bbox-widget]').forEach(widget => {
    const button = widget.querySelector('.js-bbox-zoom');
    const frame = widget.querySelector('[data-bbox]');
    const image = frame?.querySelector('img');
    if (!button || !frame || !image) return;
    const box = frame.dataset.bbox.split(',').map(Number);
    if (box.length !== 4 || !box.every(Number.isFinite) || box[2] <= 0 || box[3] <= 0) {
      button.disabled = true;
    }
    const thumbnail = image.getAttribute('src');
    const originalStyle = image.getAttribute('style');
    let zoomed = false;
    let previewScale = 1, baseHeight, baseWidth;
    const smaller = widget.querySelector('.js-preview-smaller');
    const larger = widget.querySelector('.js-preview-larger');
    function resetImageStyle() {
      if (originalStyle == null) image.removeAttribute('style');
      else image.setAttribute('style', originalStyle);
    }
    function position() {
      if (!zoomed || !image.naturalWidth || !image.naturalHeight || !frame.clientWidth || !frame.clientHeight) return;
      const scale = Math.min(frame.clientWidth / (image.naturalWidth * box[2] * 1.1), frame.clientHeight / (image.naturalHeight * box[3] * 1.1));
      const x = frame.clientWidth / 2 - image.naturalWidth * box[0] * scale;
      const y = frame.clientHeight / 2 - image.naturalHeight * box[1] * scale;
      Object.assign(image.style, {position: 'absolute', maxWidth: 'none', width: image.naturalWidth + 'px',
        height: image.naturalHeight + 'px', transform: 'translate(' + x + 'px,' + y + 'px) scale(' + scale + ')'});
      image.style.setProperty('width', image.naturalWidth + 'px', 'important');
      image.style.setProperty('height', image.naturalHeight + 'px', 'important');
    }
    function resizePreview(factor) {
      if (!baseHeight) {
        baseHeight = frame.clientHeight;
        baseWidth = frame.clientWidth;
      }
      if (!baseHeight || !baseWidth) return;
      previewScale = Math.min(3, Math.max(.5, previewScale * factor));
      frame.style.height = baseHeight * previewScale + 'px';
      const wrapper = widget.classList.contains('similar-pair-thumbnail-widget') ? widget : null;
      if (wrapper) {
        wrapper.style.width = wrapper.style.flexBasis = baseWidth * previewScale + 'px';
        frame.style.width = '100%';
      }
      if (smaller) smaller.disabled = previewScale <= .5;
      if (larger) larger.disabled = previewScale >= 3;
      position();
    }
    smaller?.addEventListener('click', () => resizePreview(.8));
    larger?.addEventListener('click', () => resizePreview(1.25));
    button.addEventListener('click', () => {
      zoomed = !zoomed;
      button.setAttribute('aria-pressed', String(zoomed));
      button.textContent = zoomed ? 'Full media' : 'Zoom to bbox';
      button.classList.toggle('active', zoomed);
      resetImageStyle();
      const url = zoomed ? frame.dataset.fullSrc || thumbnail : thumbnail;
      if (image.getAttribute('src') !== url) image.setAttribute('src', url);
      if (zoomed) position();
    });
    image.addEventListener('load', position);
    image.addEventListener('error', () => {
      if (!zoomed) return;
      zoomed = false;
      button.setAttribute('aria-pressed', 'false');
      button.textContent = 'Zoom to bbox';
      button.classList.remove('active');
      button.disabled = true;
      resetImageStyle();
      if (image.getAttribute('src') !== thumbnail) image.setAttribute('src', thumbnail);
    });
    if (window.ResizeObserver) new ResizeObserver(position).observe(frame);
    else window.addEventListener('resize', position);
  });
})();
