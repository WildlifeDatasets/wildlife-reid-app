(() => {
  'use strict';
  const form = document.getElementById('bulk-similar-merge-form');
  if (!form) return;
  const selections = Array.from(document.querySelectorAll('input[name="pairs"][form="bulk-similar-merge-form"]:not(:disabled)'));
  const directions = Array.from(document.querySelectorAll('.js-pair-direction'));
  const button = form.querySelector('button[type="submit"]');
  const status = document.getElementById('bulk-merge-selection-count');

  function updateSelection() {
    const count = selections.filter(checkbox => checkbox.checked).length;
    button.disabled = count === 0;
    if (selections.length) button.hidden = count === 0;
    status.textContent = count + (count === 1 ? ' pair selected' : ' pairs selected');
  }

  function updateDirections(save = false) {
    if (!directions.length) return;
    const reversed = directions.filter(input => input.checked).map(input => input.value).join(',');
    const current = new URL(window.location.href);
    if (reversed) current.searchParams.set('reversed', reversed);
    else current.searchParams.delete('reversed');
    const returnUrl = current.pathname + current.search + current.hash;
    if (save) history.replaceState(null, '', returnUrl);
    document.querySelectorAll('.js-pairs-return-url').forEach(input => { input.value = returnUrl; });
    const filter = document.querySelector('#similar-pairs-filter-form input[name="reversed"]');
    if (filter) filter.value = reversed;
    document.querySelectorAll('.js-pairs-layout').forEach(link => {
      const url = new URL(link.href);
      if (reversed) url.searchParams.set('reversed', reversed);
      else url.searchParams.delete('reversed');
      link.href = url.pathname + url.search;
    });
    directions.forEach(input => {
      const row = input.closest('.similar-pair-row');
      const source = input.checked ? row.dataset.nameB : row.dataset.nameA;
      const target = input.checked ? row.dataset.nameA : row.dataset.nameB;
      const description = 'Merge ' + source + ' into ' + target;
      row.querySelector('.js-pair-arrow').textContent = input.checked ? '←' : '→';
      row.querySelector('.similar-pair-row-direction label').title = description + '. Click to reverse.';
      row.querySelector('.js-pair-direction-description').textContent = description;
      row.querySelector('input[name="pairs"]').setAttribute('aria-label', 'Select merge ' + source + ' into ' + target);
      const link = row.querySelector('.js-pair-preview');
      const url = new URL(input.checked ? link.dataset.reverseUrl : link.dataset.forwardUrl, window.location.href);
      url.searchParams.set('next', returnUrl);
      link.href = url.pathname + url.search;
    });
  }

  selections.forEach(input => input.addEventListener('change', updateSelection));
  directions.forEach(input => input.addEventListener('change', () => updateDirections(true)));
  window.addEventListener('pageshow', () => { updateDirections(); updateSelection(); });
  updateDirections();
  updateSelection();
})();
