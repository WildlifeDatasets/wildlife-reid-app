document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById('mediafile-info-modal');
    if (!modal || !window.bootstrap) return;
    let copyText = '';
    let returnFocus = null;
    const triggers = '.mediafile-info-trigger';
    function preview(event) {
        const button = event.target.closest(triggers);
        if (!button || modal.classList.contains('show')) return;
        bootstrap.Tooltip.getOrCreateInstance(button, {
            title: () => button.dataset.summary, html: false, container: 'body',
            customClass: 'mediafile-info-tooltip', trigger: 'hover focus',
            delay: {show: 250, hide: 100}
        }).show();
    }
    document.addEventListener('pointerover', preview);
    document.addEventListener('focusin', preview);
    document.addEventListener('click', event => {
        const button = event.target.closest(triggers);
        if (!button) return;
        event.preventDefault();
        event.stopPropagation();
        const data = JSON.parse(button.dataset.mediafileInfo);
        bootstrap.Tooltip.getInstance(button)?.hide();
        const rows = modal.querySelector('.mediafile-info-rows');
        rows.replaceChildren();
        data.rows.forEach(row => {
            const label = document.createElement('dt'), value = document.createElement('dd');
            label.textContent = row.label; value.textContent = row.value;
            rows.append(label, value);
        });
        modal.querySelector('#mediafile-info-raw').value = data.metadata;
        modal.querySelector('#mediafile-info-detail').href = button.dataset.detailUrl;
        modal.querySelector('#mediafile-info-copy-status').textContent = '';
        copyText = data.rows.map(row => `${row.label}: ${row.value}`).join('\n') + '\n\nSource metadata:\n' + data.metadata;
        returnFocus = button;
        bootstrap.Modal.getOrCreateInstance(modal).show();
    });
    modal.addEventListener('hidden.bs.modal', () => returnFocus?.focus({preventScroll: true}));
    modal.querySelector('#mediafile-info-copy').addEventListener('click', async () => {
        const status = modal.querySelector('#mediafile-info-copy-status');
        try {
            if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(copyText);
            else {
                const input = document.createElement('textarea');
                input.value = copyText; input.style.cssText = 'position:fixed;opacity:0'; modal.append(input);
                input.select(); const copied = document.execCommand('copy'); input.remove();
                if (!copied) throw new Error('Copy unavailable');
            }
            status.textContent = 'Copied';
        } catch (_) {
            status.textContent = 'Select and copy the metadata manually.';
            modal.querySelector('#mediafile-info-raw').focus();
            modal.querySelector('#mediafile-info-raw').select();
        }
    });
});
