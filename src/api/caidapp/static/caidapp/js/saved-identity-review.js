document.addEventListener("DOMContentLoaded", () => {
    const button = document.querySelector("[data-saved-identity-review]");
    const form = button?.closest("form");
    if (!form) return;

    function savedFields() {
        return JSON.stringify(Array.from(new FormData(form), ([name, value]) => [
            name, value instanceof File ? [value.name, value.size, value.name ? value.lastModified : 0] : value,
        ]));
    }

    const initial = savedFields();
    form.addEventListener("click", event => {
        const trigger = event.target.closest("[data-saved-identity-review]");
        if (!trigger) return;
        if (savedFields() !== initial) {
            let message = trigger.parentElement.querySelector("[data-save-before-review]");
            if (!message) {
                message = document.createElement("p");
                message.dataset.saveBeforeReview = "";
                message.className = "text-warning-emphasis";
                message.setAttribute("role", "status");
                trigger.after(message);
            }
            message.textContent = "Save your changes before reviewing identity suggestions.";
            return;
        }
        window.location.assign(trigger.dataset.savedIdentityReview);
    });
});
