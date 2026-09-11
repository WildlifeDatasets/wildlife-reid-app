document.addEventListener('DOMContentLoaded', function() {
    function getCollapseTrigger(collapseElement) {
        const triggerId = collapseElement.getAttribute('id');
        if (!triggerId) {
            return null;
        }
        return document.querySelector(`[href="#${triggerId}"], [data-bs-target="#${triggerId}"]`);
    }

    function setSidebarMenuExpanded(collapseElement, expanded, persist) {
        collapseElement.classList.toggle('show', expanded);
        const triggerLink = getCollapseTrigger(collapseElement);
        if (triggerLink) {
            triggerLink.classList.toggle('collapsed', !expanded);
            triggerLink.setAttribute('aria-expanded', expanded ? 'true' : 'false');
        }
        if (persist) {
            localStorage.setItem(`sidebar-menu-${collapseElement.id}`, expanded ? 'true' : 'false');
        }
    }
    
    // 1. Sidebar Toggle Logic
    const sidebarToggle = document.getElementById('sidebarToggle');
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', () => {
            document.body.classList.toggle('sidebar-collapsed');
            
            // Optional: Save state to localStorage
            const isCollapsed = document.body.classList.contains('sidebar-collapsed');
            localStorage.setItem('sidebar-collapsed', isCollapsed);
        });
    }

    // Restore state from localStorage
    if (localStorage.getItem('sidebar-collapsed') === 'true') {
        document.body.classList.add('sidebar-collapsed');
    }

    // Restore expanded sidebar submenus.
    document.querySelectorAll('#sidebar .collapse[id]').forEach(collapseElement => {
        if (localStorage.getItem(`sidebar-menu-${collapseElement.id}`) === 'true') {
            setSidebarMenuExpanded(collapseElement, true, false);
        }

        collapseElement.addEventListener('shown.bs.collapse', () => {
            setSidebarMenuExpanded(collapseElement, true, true);
        });
        collapseElement.addEventListener('hidden.bs.collapse', () => {
            setSidebarMenuExpanded(collapseElement, false, true);
        });
    });

    // 2. Highlight Active Menu Item
    const currentPath = window.location.pathname;
    const menuLinks = document.querySelectorAll('#sidebar .nav-link, #sidebar .list-group-item');

    menuLinks.forEach(link => {
        // Simple exact match
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
            
            // If inside a submenu (collapse), open it
            const parentCollapse = link.closest('.collapse');
            if (parentCollapse) {
                setSidebarMenuExpanded(parentCollapse, true, true);
            }
        }
    });

    // Keep bbox-bearing cards static until the user explicitly explores a video.
    // The animated WebP lives in a data attribute, so it is not downloaded up front.
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    let activeMotionImage = null;

    function stopMotionPreview(image) {
        if (!image || !image.dataset.staticSrc) {
            return;
        }
        image.src = image.dataset.staticSrc;
        image.closest('.bbox-preview-frame, .observation-preview')?.classList.remove('is-motion-playing');
        if (activeMotionImage === image) {
            activeMotionImage = null;
        }
    }

    document.querySelectorAll('.js-motion-preview[data-motion-src]').forEach(image => {
        const interactionTarget = image.closest('a') || image;
        image.dataset.staticSrc = image.currentSrc || image.src;

        const startMotionPreview = () => {
            if (reduceMotion.matches) {
                return;
            }
            if (activeMotionImage && activeMotionImage !== image) {
                stopMotionPreview(activeMotionImage);
            }
            image.src = image.dataset.motionSrc;
            image.closest('.bbox-preview-frame, .observation-preview')?.classList.add('is-motion-playing');
            activeMotionImage = image;
        };

        interactionTarget.addEventListener('pointerenter', startMotionPreview);
        interactionTarget.addEventListener('pointerleave', () => stopMotionPreview(image));
        interactionTarget.addEventListener('focus', startMotionPreview);
        interactionTarget.addEventListener('blur', () => stopMotionPreview(image));
    });

    // Bounding boxes use coordinates relative to the original media file.  Card
    // previews use object-fit: contain, which can leave empty space around a
    // portrait or landscape image.  Keep the overlay's coordinate system on the
    // rendered image area rather than on the whole 4:3 preview frame.
    function syncContainedImageOverlay(frame) {
        const image = frame.querySelector('img');
        if (!image || !image.naturalWidth || !image.naturalHeight) {
            return;
        }

        const frameWidth = frame.clientWidth;
        const frameHeight = frame.clientHeight;
        if (!frameWidth || !frameHeight) {
            return;
        }

        const imageAspect = image.naturalWidth / image.naturalHeight;
        const frameAspect = frameWidth / frameHeight;
        const imageWidth = frameAspect > imageAspect ? frameHeight * imageAspect : frameWidth;
        const imageHeight = frameAspect > imageAspect ? frameHeight : frameWidth / imageAspect;
        const prefix = frame.classList.contains('observation-preview') ? '--observation-image' : '--bbox-image';

        frame.style.setProperty(`${prefix}-left`, `${(frameWidth - imageWidth) / 2}px`);
        frame.style.setProperty(`${prefix}-top`, `${(frameHeight - imageHeight) / 2}px`);
        frame.style.setProperty(`${prefix}-width`, `${imageWidth}px`);
        frame.style.setProperty(`${prefix}-height`, `${imageHeight}px`);
    }

    function syncContainedImageOverlays(root = document) {
        root.querySelectorAll('.bbox-preview-frame, .observation-preview').forEach(frame => {
            const image = frame.querySelector('img');
            if (!image) {
                return;
            }
            syncContainedImageOverlay(frame);
            if (!image.dataset.overlayLoadListener) {
                image.addEventListener('load', () => syncContainedImageOverlay(frame));
                image.dataset.overlayLoadListener = 'true';
            }
        });
    }

    syncContainedImageOverlays();
    window.addEventListener('resize', syncContainedImageOverlays);

    if ('ResizeObserver' in window) {
        const overlayResizeObserver = new ResizeObserver(entries => {
            entries.forEach(entry => syncContainedImageOverlay(entry.target));
        });
        document.querySelectorAll('.bbox-preview-frame, .observation-preview').forEach(frame => {
            overlayResizeObserver.observe(frame);
        });
    }
});

// function toggleTheme() {
//     const html = document.documentElement;
//     html.dataset.bsTheme =
//         html.dataset.bsTheme === "dark" ? "light" : "dark";
// }
function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-bs-theme') || 'light';
    const next = current === 'dark' ? 'light' : 'dark';

    html.setAttribute('data-bs-theme', next);
    localStorage.setItem('theme', next);
}


console.log("114 LAYOUT JS VERSION 789");
