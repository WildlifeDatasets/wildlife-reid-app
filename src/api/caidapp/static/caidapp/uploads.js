function checkStatuses(fetchUrl) {
    const progressElements = document.querySelectorAll('.upload-progress[data-archive-id]');
    const archiveIds = Array.from(progressElements).map(element => element.dataset.archiveId);
    const requestUrl = archiveIds.length
        ? fetchUrl + (fetchUrl.includes('?') ? '&' : '?') + 'ids=' + encodeURIComponent(archiveIds.join(','))
        : fetchUrl;

    fetch(requestUrl)
        .then(response => response.json())
        .then(data => {
            data.archives.forEach(item => {
                const statusElement = document.getElementById('status-' + item.id);
                if (statusElement && statusElement.textContent !== item.status) {
                    statusElement.textContent = item.status;
                    statusElement.className = 'badge badge-status rounded-pill bg-' + item.status_style;
                    updateTooltip(statusElement, item.status_message);
                }

                const progressElement = document.getElementById('progress-' + item.id);
                if (!progressElement) {
                    return;
                }
                if (!item.progress) {
                    progressElement.classList.add('d-none');
                    progressElement.removeAttribute('aria-valuenow');
                    return;
                }

                progressElement.classList.remove('d-none');
                const percentElement = progressElement.querySelector('.upload-progress-percent');
                if (item.progress.percent === null) {
                    percentElement.textContent = 'Processing';
                    percentElement.classList.add('visually-hidden');
                    progressElement.removeAttribute('aria-valuenow');
                } else {
                    percentElement.textContent = item.progress.percent + '%';
                    percentElement.classList.remove('visually-hidden');
                    progressElement.setAttribute('aria-valuenow', item.progress.percent);
                }
                const tooltip = item.progress.percent === null
                    ? item.progress.message
                    : item.progress.message + ': ' + item.progress.percent + '%';
                updateTooltip(progressElement, tooltip);
            });

            setTimeout(checkStatuses, 10000, fetchUrl);
        })
        .catch(error => {
            console.error(error);
            setTimeout(checkStatuses, 20000, fetchUrl);
        });
}

function updateTooltip(element, message) {
    element.setAttribute('title', message || '');
    element.setAttribute('data-bs-original-title', message || '');
    if (window.bootstrap && bootstrap.Tooltip) {
        const tooltip = bootstrap.Tooltip.getInstance(element);
        if (tooltip) {
            tooltip.dispose();
            new bootstrap.Tooltip(element);
        }
    }
}
