document.addEventListener('DOMContentLoaded', function() {
    
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
                parentCollapse.classList.add('show');
                // Highlight parent trigger
                const triggerId = parentCollapse.getAttribute('id');
                const triggerLink = document.querySelector(`[data-bs-target="#${triggerId}"]`);
                if (triggerLink) {
                    triggerLink.classList.remove('collapsed');
                    triggerLink.setAttribute('aria-expanded', 'true');
                    triggerLink.classList.add('active'); // Optional: highlight parent too
                }
            }
        }
    });
});

function toggleTheme() {
    const html = document.documentElement;
    html.dataset.bsTheme =
        html.dataset.bsTheme === "dark" ? "light" : "dark";
}

console.log("114 LAYOUT JS VERSION 789");