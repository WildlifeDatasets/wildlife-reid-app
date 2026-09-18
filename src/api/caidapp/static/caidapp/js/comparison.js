/* Two independent, read-only comparison panels. */
(() => {
  'use strict';
  const app = document.getElementById('comparison-app');
  if (!app) return;
  const kinds = {observation: 'Observation', mediafile: 'Media file', identity: 'Identity', album: 'Album'};
  const panels = {};
  const pageSize = 24;
  let syncingTray = false;
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
  const query = (p, selector) => p.el.querySelector(selector);
  const selected = p => p.items.find(item => String(item.id) === String(p.itemId));
  const sameSource = (a, b) => (!a && !b) || (a && b && a.kind === b.kind && String(a.id) === String(b.id));

  function saveURL(p) {
    const params = new URLSearchParams(location.search);
    for (const key of ['kind', 'id', 'mode', 'page', 'item']) params.delete(p.slot + '_' + key);
    if (p.source) {
      const values = {kind: p.source.kind, id: p.source.id, mode: p.mode, page: p.page, item: p.itemId};
      for (const [key, value] of Object.entries(values)) {
        if (value != null) params.set(p.slot + '_' + key, value);
      }
    }
    history.replaceState(null, '', location.pathname + (params.size ? '?' + params : '') + location.hash);
  }

  function syncTray(p) {
    if (!window.ComparisonTray) return;
    syncingTray = true;
    try { window.ComparisonTray.setSlot(p.slot, p.source); }
    finally { syncingTray = false; }
  }

  function resetView(p) {
    p.view = 'full';
    p.zoom = 1;
    p.panX = p.panY = 0;
    p.drag = null;
  }

  function stopLoad(p) {
    p.abort?.abort();
    p.generation++;
  }

  function setSource(p, source, user = false) {
    stopLoad(p);
    p.source = source ? {kind: source.kind, id: String(source.id), label: source.label || ''} : null;
    p.mode = 'mediafiles';
    p.page = 1;
    p.itemId = null;
    p.items = [];
    p.data = null;
    p.error = '';
    p.loading = false;
    resetView(p);
    saveURL(p);
    if (user) syncTray(p);
    if (p.source) load(p);
    else render(p);
  }

  async function load(p, preferLast = false) {
    stopLoad(p);
    const generation = p.generation;
    const desiredItem = p.itemId;
    p.items = [];
    p.data = null;
    p.error = '';
    p.loading = true;
    resetView(p);
    render(p);
    p.abort = new AbortController();
    const params = new URLSearchParams({kind: p.source.kind, id: p.source.id, mode: p.mode, page: p.page});
    try {
      const response = await fetch(app.dataset.sourceUrl + '?' + params, {
        signal: p.abort.signal, headers: {Accept: 'application/json'}
      });
      if (!response.ok) throw new Error(response.status === 404
        ? 'This source or page is unavailable.' : 'Could not load this source. Please try again.');
      const data = await response.json();
      if (generation !== p.generation) return;
      p.data = data;
      p.items = data.items || [];
      p.page = data.page;
      p.source.label = data.source.label;
      const item = p.items.find(i => String(i.id) === String(desiredItem))
        || (preferLast ? p.items.at(-1) : p.items[0]);
      p.itemId = item ? String(item.id) : null;
      p.loading = false;
      saveURL(p);
      syncTray(p);
      render(p);
    } catch (error) {
      if (error.name === 'AbortError' || generation !== p.generation) return;
      p.loading = false;
      p.itemId = null;
      p.error = error instanceof SyntaxError ? 'Session expired or invalid response. Reload and sign in again.' : error.message;
      render(p);
    }
  }

  function markup(slot) {
    return '<h2 class="h5">Side ' + slot.toUpperCase() + '</h2>'
      + '<div class="comparison-source-form"><select class="form-select form-select-sm js-kind" aria-label="Source type ' + slot.toUpperCase() + '">'
      + Object.entries(kinds).map(([kind, label]) => '<option value="' + kind + '">' + label + '</option>').join('')
      + '</select><div class="comparison-search-wrap"><input class="form-control form-control-sm js-search" autocomplete="off" aria-label="Search source ' + slot.toUpperCase() + '" placeholder="Search name, code, filename or ID">'
      + '<div class="comparison-search-results d-none" aria-label="Search results"></div></div></div>'
      + '<div class="comparison-source-meta mt-2 small"></div>'
      + '<div class="comparison-mode mt-2"><label for="comparison-mode-' + slot + '" class="small me-2">Album contents</label>'
      + '<select id="comparison-mode-' + slot + '" class="form-select form-select-sm d-inline-block w-auto js-mode"><option value="mediafiles">Media files</option><option value="observations">Observations</option></select></div>'
      + '<div class="comparison-status small mt-2" role="status" aria-live="polite"></div>'
      + '<div class="comparison-viewer mt-3" tabindex="0" aria-label="Media viewer ' + slot.toUpperCase() + '"></div>'
      + '<div class="comparison-viewer-controls mt-2"><div class="btn-group btn-group-sm">'
      + '<button type="button" class="btn btn-outline-secondary js-full">Full media</button>'
      + '<button type="button" class="btn btn-outline-secondary js-crop">BBox zoom</button>'
      + '<button type="button" class="btn btn-outline-secondary js-zoom-out" aria-label="Zoom out">−</button>'
      + '<button type="button" class="btn btn-outline-secondary js-zoom-in" aria-label="Zoom in">+</button>'
      + '<button type="button" class="btn btn-outline-secondary js-reset">Reset view</button></div>'
      + '<div class="dropdown"><button type="button" class="btn btn-sm btn-outline-primary dropdown-toggle js-item-actions-toggle" data-bs-toggle="dropdown" aria-expanded="false" disabled>Actions</button>'
      + '<ul class="dropdown-menu dropdown-menu-end js-item-actions"></ul></div></div>'
      + '<div class="comparison-item-meta small mt-2"></div>'
      + '<div class="comparison-pagination mt-2"></div><div class="comparison-gallery mt-2"></div>';
  }

  function render(p) {
    p.el.dataset.kind = p.source?.kind || '';
    if (p.source) query(p, '.js-kind').value = p.source.kind;
    query(p, '.js-mode').value = p.mode;
    query(p, '.comparison-source-meta').innerHTML = p.data
      ? '<a href="' + escape(p.data.source.detail_url) + '">' + escape(p.data.source.label) + '</a>'
      : escape(p.source?.label || (p.source ? kinds[p.source.kind] + ' ' + p.source.id : 'Choose a source, or add items while browsing.'));
    const status = query(p, '.comparison-status');
    status.classList.toggle('text-danger', !!p.error);
    status.textContent = p.error || (p.loading ? 'Loading…' : p.data ? p.data.count + ' items' : '');
    if (p.error) {
      const retry = document.createElement('button');
      retry.className = 'btn btn-sm btn-outline-secondary ms-2';
      retry.type = 'button';
      retry.textContent = 'Retry';
      retry.onclick = () => load(p);
      status.append(retry);
    }
    query(p, '.comparison-gallery').innerHTML = p.items.map(item =>
      '<button type="button" class="comparison-thumb ' + (String(item.id) === String(p.itemId) ? 'is-selected' : '')
      + '" data-item="' + escape(item.id) + '" title="' + escape(item.label) + '" aria-label="' + escape(item.label)
      + '" aria-pressed="' + (String(item.id) === String(p.itemId)) + '">'
      + (item.thumbnail_url ? '<img loading="lazy" src="' + escape(item.thumbnail_url) + '" alt="">' : '<span>No preview</span>')
      + (item.media_type === 'video' ? '<span class="badge text-bg-dark">Video</span>' : '') + '</button>'
    ).join('');
    renderNavigation(p);
    renderViewer(p);
  }

  function renderNavigation(p) {
    const container = query(p, '.comparison-pagination');
    if (!p.data || !p.items.length) { container.innerHTML = ''; return; }
    const index = p.items.findIndex(item => String(item.id) === String(p.itemId));
    const first = p.page === 1 && index === 0;
    const last = p.page === p.data.num_pages && index === p.items.length - 1;
    container.innerHTML =
      '<button type="button" class="btn btn-sm btn-outline-secondary js-prev-item" ' + (first ? 'disabled' : '') + '>‹ Previous item</button>'
      + '<span class="small">' + ((p.page - 1) * pageSize + index + 1) + ' / ' + p.data.count + '</span>'
      + '<button type="button" class="btn btn-sm btn-outline-secondary js-next-item" ' + (last ? 'disabled' : '') + '>Next item ›</button>';
    query(p, '.js-prev-item').onclick = () => moveItem(p, -1);
    query(p, '.js-next-item').onclick = () => moveItem(p, 1);
  }

  function chooseItem(p, id) {
    p.itemId = String(id);
    resetView(p);
    saveURL(p);
    render(p);
  }

  function changePage(p, page, preferLast = false) {
    if (p.loading || !p.data || page < 1 || page > p.data.num_pages) return;
    p.page = page;
    p.itemId = null;
    load(p, preferLast);
  }

  function moveItem(p, direction) {
    if (p.loading || !p.items.length) return;
    const index = p.items.findIndex(item => String(item.id) === String(p.itemId)) + direction;
    if (index >= 0 && index < p.items.length) chooseItem(p, p.items[index].id);
    else changePage(p, p.page + direction, direction < 0);
  }

  function renderViewer(p) {
    const item = selected(p);
    const viewer = query(p, '.comparison-viewer');
    const video = item?.media_type === 'video';
    const validBox = item?.bbox && item.bbox.length === 4 && item.bbox.every(Number.isFinite)
      && item.bbox[2] > 0 && item.bbox[3] > 0;
    query(p, '.js-crop').disabled = !item || video || !validBox;
    query(p, '.js-crop').classList.toggle('active', p.view === 'crop');
    for (const control of ['.js-reset', '.js-zoom-in', '.js-zoom-out']) query(p, control).disabled = !item || video;
    query(p, '.js-full').disabled = !item;
    const actionsToggle = query(p, '.js-item-actions-toggle');
    const actionsMenu = query(p, '.js-item-actions');
    const actions = item ? [
      item.observation_edit_url && ['Edit observation', item.observation_edit_url],
      item.mediafile_edit_url && ['Edit media file', item.mediafile_edit_url],
      !item.mediafile_edit_url && item.mediafile_url && ['View media file', item.mediafile_url],
      item.identity_url && ['Edit identity', item.identity_url],
      item.locality_url && ['Edit locality', item.locality_url]
    ].filter(Boolean) : [];
    actionsToggle.disabled = !actions.length;
    actionsMenu.innerHTML = actions.map(action => '<li><a class="dropdown-item" href="' + escape(action[1]) + '">'
      + escape(action[0]) + '</a></li>').join('');
    query(p, '.comparison-item-meta').textContent = '';
    if (!item) {
      viewer.innerHTML = '<div class="comparison-viewer-empty">' + (p.loading ? 'Loading media…' : p.error
        ? 'No media to display.' : p.source ? 'No items in this source.' : 'Add an item or choose a source above.') + '</div>';
      return;
    }
    const url = video ? item.media_url : item.image_url || item.media_url;
    if (!url) viewer.innerHTML = '<div class="comparison-viewer-empty">Media file unavailable.</div>';
    else viewer.innerHTML = video
      ? '<video controls preload="metadata" src="' + escape(url) + '"></video>'
      : '<img draggable="false" src="' + escape(url) + '" alt="' + escape(item.label) + '">'
        + (validBox && p.view === 'full' ? '<span class="comparison-bbox"></span>' : '');
    query(p, '.comparison-item-meta').innerHTML =
      (item.detail_url ? '<a href="' + escape(item.detail_url) + '">' + escape(item.label) + '</a>' : escape(item.label))
      + (item.identity_label ? ' · Identity: ' + (item.identity_url ? '<a href="' + escape(item.identity_url) + '">'
        + escape(item.identity_label) + '</a>' : escape(item.identity_label)) : '')
      + (item.is_placeholder ? '<span class="d-block">No-detection placeholder</span>' : '');
    const image = viewer.querySelector('img');
    if (image) {
      image.onload = () => positionImage(p);
      image.onerror = () => { viewer.innerHTML = '<div class="comparison-viewer-empty">Media file could not be loaded.</div>'; };
      if (image.complete) positionImage(p);
    }
  }

  function positionImage(p) {
    const viewer = query(p, '.comparison-viewer'), image = viewer.querySelector('img'), item = selected(p);
    if (!image || !item || !image.naturalWidth || !image.naturalHeight) return;
    const width = viewer.clientWidth, height = viewer.clientHeight;
    const naturalWidth = image.naturalWidth, naturalHeight = image.naturalHeight;
    let scale = Math.min(width / naturalWidth, height / naturalHeight) * p.zoom;
    let x = (width - naturalWidth * scale) / 2 + p.panX;
    let y = (height - naturalHeight * scale) / 2 + p.panY;
    if (p.view === 'crop' && item.bbox) {
      const box = item.bbox;
      scale = Math.min(width / (naturalWidth * box[2] * 1.1), height / (naturalHeight * box[3] * 1.1)) * p.zoom;
      x = width / 2 - naturalWidth * scale * box[0] + p.panX;
      y = height / 2 - naturalHeight * scale * box[1] + p.panY;
    }
    image.style.width = naturalWidth + 'px';
    image.style.height = naturalHeight + 'px';
    image.style.transform = 'translate(' + x + 'px,' + y + 'px) scale(' + scale + ')';
    const overlay = viewer.querySelector('.comparison-bbox');
    if (overlay) {
      const box = item.bbox;
      Object.assign(overlay.style, {
        left: (x + naturalWidth * scale * (box[0] - box[2] / 2)) + 'px',
        top: (y + naturalHeight * scale * (box[1] - box[3] / 2)) + 'px',
        width: naturalWidth * scale * box[2] + 'px', height: naturalHeight * scale * box[3] + 'px'
      });
    }
  }

  function wireSearch(p) {
    const input = query(p, '.js-search'), results = query(p, '.comparison-search-results');
    let timer, generation = 0, abort;
    function cancel() { clearTimeout(timer); generation++; abort?.abort(); results.classList.add('d-none'); }
    input.addEventListener('input', () => {
      cancel();
      const value = input.value.trim();
      if (!value) return;
      const current = generation;
      timer = setTimeout(async () => {
        abort = new AbortController();
        try {
          const response = await fetch(app.dataset.searchUrl + '?' + new URLSearchParams({
            kind: query(p, '.js-kind').value, q: value
          }), {signal: abort.signal, headers: {Accept: 'application/json'}});
          if (!response.ok) throw new Error('Search unavailable.');
          const data = await response.json();
          if (current !== generation) return;
          results.innerHTML = (data.results || []).map(source => '<button type="button" data-kind="' + escape(source.kind)
            + '" data-id="' + escape(source.id) + '">' + escape(source.label) + '</button>').join('')
            || '<div class="p-2 small">No matching sources.</div>';
          results.classList.remove('d-none');
        } catch (error) {
          if (error.name === 'AbortError' || current !== generation) return;
          results.textContent = 'Search unavailable. Try again.';
          results.classList.remove('d-none');
        }
      }, 200);
    });
    results.addEventListener('click', event => {
      const button = event.target.closest('button[data-id]');
      if (!button) return;
      cancel();
      input.value = '';
      setSource(p, {kind: button.dataset.kind, id: button.dataset.id, label: button.textContent}, true);
    });
    query(p, '.js-kind').addEventListener('change', () => {
      cancel();
      input.value = '';
      setSource(p, null, true);
      input.focus();
    });
    input.addEventListener('keydown', event => { if (event.key === 'Escape') cancel(); });
    document.addEventListener('click', event => { if (!p.el.contains(event.target)) cancel(); });
  }

  function makePanel(slot) {
    const el = app.querySelector('[data-slot="' + slot + '"]');
    el.innerHTML = markup(slot);
    const p = {el, slot, source: null, mode: 'mediafiles', page: 1, itemId: null, items: [], data: null,
      generation: 0, error: '', loading: false};
    resetView(p);
    wireSearch(p);
    query(p, '.js-mode').addEventListener('change', event => {
      if (!p.source) return;
      p.mode = event.target.value;
      p.page = 1;
      p.itemId = null;
      load(p);
    });
    query(p, '.comparison-gallery').addEventListener('click', event => {
      const button = event.target.closest('[data-item]');
      if (button) chooseItem(p, button.dataset.item);
    });
    query(p, '.js-full').onclick = query(p, '.js-reset').onclick = () => {resetView(p); renderViewer(p);};
    query(p, '.js-crop').onclick = () => {resetView(p); p.view = 'crop'; renderViewer(p);};
    function zoom(factor) {p.zoom = Math.min(12, Math.max(.5, p.zoom * factor)); positionImage(p);}
    query(p, '.js-zoom-in').onclick = () => zoom(1.25);
    query(p, '.js-zoom-out').onclick = () => zoom(.8);
    const viewer = query(p, '.comparison-viewer');
    viewer.addEventListener('wheel', event => {
      if (!viewer.querySelector('img')) return;
      event.preventDefault();
      zoom(event.deltaY < 0 ? 1.12 : 1 / 1.12);
    }, {passive: false});
    viewer.addEventListener('pointerdown', event => {
      if (!viewer.querySelector('img')) return;
      viewer.focus();
      p.drag = {x: event.clientX, y: event.clientY};
      viewer.setPointerCapture(event.pointerId);
    });
    viewer.addEventListener('pointermove', event => {
      if (!p.drag) return;
      p.panX += event.clientX - p.drag.x;
      p.panY += event.clientY - p.drag.y;
      p.drag = {x: event.clientX, y: event.clientY};
      positionImage(p);
    });
    for (const event of ['pointerup', 'pointercancel', 'lostpointercapture']) viewer.addEventListener(event, () => {p.drag = null;});
    el.addEventListener('keydown', event => {
      if (event.target.matches('input,select,textarea,video')) return;
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        event.preventDefault();
        moveItem(p, event.key === 'ArrowLeft' ? -1 : 1);
      }
    });
    if (window.ResizeObserver) new ResizeObserver(() => positionImage(p)).observe(viewer);
    else window.addEventListener('resize', () => positionImage(p));
    return p;
  }

  // Capture both URL sources before either panel writes back its state.
  const params = new URLSearchParams(location.search);
  const tray = window.ComparisonTray?.getState() || {};
  for (const slot of ['a', 'b']) {
    const p = panels[slot] = makePanel(slot);
    const id = params.get(slot + '_id'), kind = params.get(slot + '_kind');
    if (id && kinds[kind]) {
      p.source = {kind, id, label: ''};
      p.mode = params.get(slot + '_mode') === 'observations' ? 'observations' : 'mediafiles';
      const page = Number(params.get(slot + '_page'));
      p.page = Number.isInteger(page) && page > 0 ? page : 1;
      p.itemId = params.get(slot + '_item');
      syncTray(p);
      load(p);
    } else if (tray[slot]) setSource(p, tray[slot]);
    else render(p);
  }
  document.addEventListener('comparison:change', event => {
    if (syncingTray) return;
    for (const slot of ['a', 'b']) {
      const incoming = event.detail?.[slot] || null;
      if (!sameSource(incoming, panels[slot].source)) setSource(panels[slot], incoming);
    }
  });
  app.querySelector('.js-comparison-fullscreen').addEventListener('click', async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else if (app.requestFullscreen) await app.requestFullscreen();
    } catch (_) { /* Full-screen mode is optional in embedded browsers. */ }
  });
})();
