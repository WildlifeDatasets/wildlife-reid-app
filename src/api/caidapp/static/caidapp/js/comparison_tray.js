/* Shared comparison selection. No domain data is changed by this module. */
(() => {
  'use strict';
  const tray = document.getElementById('comparison-tray');
  if (!tray) return;
  const kinds = new Set(['observation', 'mediafile', 'identity', 'album']);
  const key = tray.dataset.storageKey;
  const dialog = document.getElementById('comparison-replace-dialog');
  const status = document.getElementById('comparison-tray-status');
  let pending = null;
  let state = {a: null, b: null};
  function clean(source) {
    if (!source || !kinds.has(source.kind) || !String(source.id || '').trim()) return null;
    let label = String(source.label || `${source.kind} ${source.id}`);
    if (source.kind === 'observation' && !label.startsWith('Observation ')) label = `Observation ${source.id} · ${label}`;
    return {kind: source.kind, id: String(source.id), label};
  }
  try {
    const saved = JSON.parse(sessionStorage.getItem(key));
    if (saved) state = {a: clean(saved.a), b: clean(saved.b)};
  } catch (_) { /* Storage may be unavailable; this page still works. */ }
  const previewCache = new Map();
  function snapshot() { return JSON.parse(JSON.stringify(state)); }
  function render() {
    tray.hidden = !state.a && !state.b;
    document.body.classList.toggle('comparison-tray-visible', !tray.hidden);
    const selectedCount = Number(Boolean(state.a)) + Number(Boolean(state.b));
    document.getElementById('comparison-tray-title').textContent = `Comparison ${selectedCount} / 2`;
    for (const slot of ['a', 'b']) {
      const source = state[slot];
      const node = tray.querySelector(`[data-slot="${slot}"]`);
      node.querySelector('.comparison-tray-label').textContent = source ? source.label : 'Empty';
      node.querySelector('.comparison-tray-label').title = source ? `${source.kind}: ${source.label}` : 'Empty';
      node.querySelector('button').disabled = !source;
      const img = node.querySelector('img');
      img.hidden = true;
      img.removeAttribute('src');
      if (source) {
        const sourceKey = `${source.kind}:${source.id}`;
        if (!previewCache.has(sourceKey)) {
          previewCache.set(sourceKey, null);
          const url = new URL(tray.dataset.sourceUrl, location.origin);
          url.search = new URLSearchParams({kind: source.kind, id: source.id});
          fetch(url, {headers: {'Accept': 'application/json'}}).then(r => r.ok ? r.json() : null).then(data => {
            previewCache.set(sourceKey, data?.items?.[0]?.thumbnail_url || data?.items?.[0]?.image_url || '');
            render();
          }).catch(() => {});
        }
        const preview = previewCache.get(sourceKey);
        if (preview) { img.src = preview; img.hidden = false; }
      }
    }
    const link = document.getElementById('comparison-tray-open');
    const ready = !!(state.a && state.b);
    link.classList.toggle('disabled', !ready);
    link.setAttribute('aria-disabled', String(!ready));
    link.tabIndex = ready ? 0 : -1;
    const url = new URL(tray.dataset.url, location.origin);
    for (const slot of ['a', 'b']) if (state[slot]) {
      url.searchParams.set(`${slot}_kind`, state[slot].kind);
      url.searchParams.set(`${slot}_id`, state[slot].id);
    }
    link.href = url.pathname + url.search;
    if (!tray.hidden) document.body.style.setProperty('--comparison-tray-height', `${tray.offsetHeight + 20}px`);
  }
  function commit(message) {
    try { sessionStorage.setItem(key, JSON.stringify(state)); } catch (_) {}
    render();
    status.textContent = message || '';
    document.dispatchEvent(new CustomEvent('comparison:change', {detail: snapshot()}));
  }
  function setSlot(slot, source) {
    if (!['a', 'b'].includes(slot)) return;
    state[slot] = clean(source);
    const selectedCount = Number(Boolean(state.a)) + Number(Boolean(state.b));
    const message = selectedCount === 2
      ? 'Two items are ready to compare.'
      : selectedCount === 1 ? 'Choose one more item to compare.' : 'Selection cleared.';
    commit(message);
  }
  function add(source) {
    source = clean(source);
    if (!source) return;
    if (!state.a) return setSlot('a', source);
    if (!state.b) return setSlot('b', source);
    pending = source;
    document.getElementById('comparison-replace-label').textContent = source.label;
    for (const slot of ['a', 'b']) {
      dialog.querySelector(`[data-comparison-replace="${slot}"]`).textContent = `Replace ${slot.toUpperCase()}: ${state[slot].label}`;
    }
    dialog.showModal();
  }
  document.addEventListener('click', event => {
    const addButton = event.target.closest('.js-comparison-add');
    if (addButton) {
      event.preventDefault();
      add({kind: addButton.dataset.comparisonKind, id: addButton.dataset.comparisonId, label: addButton.dataset.comparisonLabel});
    }
    const remove = event.target.closest('[data-comparison-remove]');
    if (remove) setSlot(remove.dataset.comparisonRemove, null);
    const replace = event.target.closest('[data-comparison-replace]');
    if (replace && pending) { setSlot(replace.dataset.comparisonReplace, pending); pending = null; dialog.close(); }
  });
  document.getElementById('comparison-replace-cancel').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => { pending = null; });
  document.getElementById('comparison-tray-clear').addEventListener('click', () => {state.a = null; state.b = null; commit('Selection cleared.');});
  const selectedButton = document.getElementById('comparison-selected-observations');
  if (selectedButton) {
    function updateSelectedObservationAction() {
      const selected = document.querySelectorAll('.js-observation-checkbox:checked').length;
      const allFiltered = document.getElementById('observation-select-all-filtered');
      const ready = selected === 2 && !allFiltered?.checked;
      selectedButton.disabled = !ready;
      selectedButton.title = ready ? 'Open the two selected observations in comparison.' : 'Select exactly two observations on this page.';
    }
    document.addEventListener('change', function(event) {
      if (event.target.matches('.js-observation-checkbox, #observation-select-all-filtered, #observation-select-page')) {
        window.setTimeout(updateSelectedObservationAction, 0);
      }
    });
    updateSelectedObservationAction();
    selectedButton.addEventListener('click', () => {
      const selected = [...document.querySelectorAll('.js-observation-checkbox:checked')];
      const allFiltered = document.getElementById('observation-select-all-filtered');
      if (selected.length !== 2 || allFiltered?.checked) {
        document.getElementById('comparison-selection-message').textContent = 'Select exactly two individual observations on this page to compare.';
        return;
      }
      for (const [index, slot] of ['a', 'b'].entries()) {
        state[slot] = clean({kind:'observation', id:selected[index].value, label:`Observation ${selected[index].value}`});
      }
      commit();
      location.assign(document.getElementById('comparison-tray-open').href);
    });
  }
  window.addEventListener('resize', render);
  window.ComparisonTray = {getState: snapshot, setSlot, add};
  render();
})();
