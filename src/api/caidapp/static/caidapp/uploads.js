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
                const statusText = statusElement?.querySelector('.badge-status-text');
                if (statusElement && statusText && statusText.textContent !== item.status) {
                    statusText.textContent = item.status;
                    Array.from(statusElement.classList)
                        .filter(className => className.startsWith('bg-'))
                        .forEach(className => statusElement.classList.remove(className));
                    statusElement.classList.add('bg-' + item.status_style);
                    updateTooltip(statusElement, item.status_message);
                }

                const progressElement = document.getElementById('progress-' + item.id);
                if (!progressElement) {
                    return;
                }
                if (!item.progress) {
                    progressElement.classList.add('d-none');
                    progressElement.removeAttribute('aria-valuenow');
                    progressElement.querySelector('.upload-progress-value').style.strokeDasharray = '0 100';
                    return;
                }

                progressElement.classList.remove('d-none');
                const progressValue = progressElement.querySelector('.upload-progress-value');
                if (item.progress.percent === null) {
                    progressValue.style.strokeDasharray = '0 100';
                    progressElement.removeAttribute('aria-valuenow');
                } else {
                    progressValue.style.strokeDasharray = item.progress.percent + ' 100';
                    progressElement.setAttribute('aria-valuenow', item.progress.percent);
                }
                progressElement.setAttribute('aria-valuetext', item.progress.message);
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
