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
            triggerLink.classList.toggle('active', expanded);
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
