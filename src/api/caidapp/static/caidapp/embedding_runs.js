(function () {
  "use strict";

  function requestKey() {
    if (window.crypto && window.crypto.randomUUID) return window.crypto.randomUUID();
    return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, function (c) {
      return (Number(c) ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> Number(c) / 4).toString(16);
    });
  }

  document.querySelectorAll("[data-embedding-start]").forEach(function (button) {
    const form = button.closest("form");
    function formSnapshot() {
      return form ? JSON.stringify(Array.from(new FormData(form), function (entry) {
        const value = entry[1];
        return [entry[0], value instanceof File ? [value.name, value.size, value.name ? value.lastModified : 0] : value];
      })) : "";
    }
    const initialSnapshot = button.dataset.checkUnsaved ? formSnapshot() : "";
    let dirty = false;
    if (button.dataset.checkUnsaved && form) {
      form.addEventListener("change", function () { dirty = true; });
      form.addEventListener("input", function () { dirty = true; });
    }
    function updateScopeLabel() {
      const label = button.querySelector("[data-embedding-label]");
      if (!label || !form) return;
      const filtered = form.querySelector("[name=select_all_filtered]");
      const selected = form.querySelector(".js-observation-checkbox:checked:not(:disabled)");
      label.textContent = selected && !(filtered && filtered.checked)
        ? "Generate embeddings for selected observations"
        : "Generate embeddings for all filtered observations";
    }
    if (button.dataset.emptyScope && form) {
      form.addEventListener("change", updateScopeLabel);
      form.addEventListener("click", function () { queueMicrotask(updateScopeLabel); });
      updateScopeLabel();
    }
    button.addEventListener("click", async function () {
      const message = document.getElementById("embedding-start-message");
      function show(text) {
        if (message) { message.textContent = text; message.hidden = false; }
      }
      if (dirty || (button.dataset.checkUnsaved && formSnapshot() !== initialSnapshot)) { show("Save your changes before generating embeddings. The task uses saved observations."); return; }
      let scope = button.dataset.scope;
      let ids = button.dataset.targetId ? [button.dataset.targetId] : [];
      if (!ids.length && button.dataset.targetIds) ids = button.dataset.targetIds.split(",").filter(Boolean);
      if (!ids.length && form) {
        ids = Array.from(form.querySelectorAll(".js-observation-checkbox:checked:not(:disabled)"), function (checkbox) { return checkbox.value; });
        const allFiltered = form.querySelector("[name=select_all_filtered]");
        if (allFiltered && allFiltered.checked) {
          if (!button.dataset.filteredScope) {
            show("Select observations on this page explicitly before generating embeddings."); return;
          }
          scope = button.dataset.filteredScope;
          ids = [];
        }
      }
      if (!ids.length && scope === button.dataset.scope && button.dataset.emptyScope) scope = button.dataset.emptyScope;
      if (!ids.length && scope !== button.dataset.emptyScope && scope !== button.dataset.filteredScope) {
        show("Select at least one observation on this page."); return;
      }
      const body = new FormData();
      body.append("scope", scope);
      ids.forEach(function (id) { body.append(scope === "mediafiles" ? "mediafile_ids" : "observation_ids", id); });
      const url = new URL(button.dataset.embeddingStart, window.location.href);
      if (scope === button.dataset.filteredScope) url.search = button.dataset.filterQuery || "";
      const selection = JSON.stringify([scope, ids.slice().sort(), url.search]);
      if (button.dataset.requestSelection !== selection) delete button.dataset.requestKey;
      button.dataset.requestSelection = selection;
      if (!button.dataset.requestKey) button.dataset.requestKey = requestKey();
      body.append("request_key", button.dataset.requestKey);
      const csrf = document.querySelector("[name=csrfmiddlewaretoken]");
      if (csrf) body.append("csrfmiddlewaretoken", csrf.value);
      button.disabled = true;
      show("Submitting embedding task…");
      try {
        const response = await fetch(url, {method: "POST", body: body, headers: {Accept: "application/json"}, credentials: "same-origin"});
        const data = await response.json();
        if (!response.ok) {
          // A rejected request can be corrected; an uncertain network result keeps its key.
          if (response.status < 500) delete button.dataset.requestKey;
          throw new Error(data.message || "The embedding task could not be submitted.");
        }
        window.location.assign(data.detail_url);
      } catch (error) {
        show(error.message || "The task could not be submitted. Try again to check the same request.");
        button.disabled = false;
      }
    });
  });
})();
