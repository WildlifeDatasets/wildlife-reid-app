(function () {
  "use strict";
  const modal = document.getElementById("identity-candidates-modal");
  if (!modal) return;
  const status = modal.querySelector("[data-candidate-status]");
  const query = modal.querySelector("[data-candidate-query]");
  const actions = modal.querySelector("[data-candidate-actions]");
  const results = modal.querySelector("[data-candidate-results]");
  let generation = 0;
  let timer;
  let trigger;
  let payload;
  let busy = false;

  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function csrf() {
    const field = document.querySelector("[name=csrfmiddlewaretoken]");
    return field ? field.value : "";
  }
  async function json(url, options) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.message || "The request failed. Try again.");
    return data;
  }
  function post(body) {
    return {method: "POST", headers: {"X-CSRFToken": csrf()}, body};
  }
  function requestKey() {
    if (window.crypto.randomUUID) return window.crypto.randomUUID();
    return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, c =>
      (Number(c) ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> Number(c) / 4).toString(16));
  }
  function editorForm() {
    return trigger && trigger.dataset.editor ? trigger.closest(".observation-form") : null;
  }
  function savedInputMatches(item) {
    const form = editorForm();
    if (!form) return true;
    const names = ["bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height"];
    return names.every((name, index) => {
      const field = form.querySelector(`[name$="-${name}"]`);
      const value = field && field.value !== "" ? Number(field.value) : null;
      return value === (item.bbox ? item.bbox[index] : null);
    }) && Array.from(form.querySelectorAll("input, select")).every(field => {
      if (field.name.endsWith("-orientation") || field.name.endsWith("-taxon")) {
        return Array.from(field.options).every(option => option.selected === option.defaultSelected);
      }
      return !field.name.endsWith("-DELETE") || !field.checked;
    });
  }
  function preview(item) {
    const link = element("a");
    link.href = item.detail_url;
    link.target = "_blank";
    link.rel = "noopener";
    link.title = "Open media file in a new tab";
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 400;
    canvas.style.cssText = "display:block;width:100%;height:auto;background:var(--wrid-media-thumb-bg,#eee)";
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", item.label);
    link.append(canvas);
    const image = new Image();
    image.onload = () => {
      const bbox = item.bbox || [0.5, 0.5, 1, 1];
      const left = Math.max(0, (bbox[0] - bbox[2] / 2) * image.naturalWidth);
      const top = Math.max(0, (bbox[1] - bbox[3] / 2) * image.naturalHeight);
      const right = Math.min(image.naturalWidth, (bbox[0] + bbox[2] / 2) * image.naturalWidth);
      const bottom = Math.min(image.naturalHeight, (bbox[1] + bbox[3] / 2) * image.naturalHeight);
      const width = right - left;
      const height = bottom - top;
      if (width <= 0 || height <= 0) return;
      const scale = Math.min(canvas.width / width, canvas.height / height);
      canvas.getContext("2d").drawImage(image, left, top, width, height,
        (canvas.width - width * scale) / 2, (canvas.height - height * scale) / 2, width * scale, height * scale);
    };
    image.onerror = () => { link.replaceChildren(element("span", "Preview unavailable — open media file")); };
    image.src = item.image_url;
    return link;
  }
  async function accept(candidate, token) {
    if (busy || token !== generation) return;
    const form = editorForm();
    if (form) {
      if (!savedInputMatches(payload.observation)) {
        status.textContent = "Save your annotation changes and reopen suggestions."; return;
      }
      const field = form.querySelector("select[name$='-identity']");
      if (!field) return;
      let option = Array.from(field.options).find(item => item.value === String(candidate.identity_id));
      if (!option) { option = new Option(candidate.label, candidate.identity_id); field.add(option); }
      field.value = String(candidate.identity_id);
      if (window.jQuery) window.jQuery(field).trigger("change");
      else field.dispatchEvent(new Event("change", {bubbles: true}));
      const representative = form.querySelector("[name$='-identity_is_representative']");
      if (representative && candidate.identity_id !== payload.expected_identity_id) representative.checked = false;
      status.textContent = "Identity selected. Save the media file to confirm your changes.";
      results.replaceChildren();
      return;
    }
    busy = true;
    results.querySelectorAll("button").forEach(button => { button.disabled = true; });
    try {
      const body = new FormData();
      body.set("identity_id", candidate.identity_id);
      body.set("expected_identity_id", payload.expected_identity_id || "");
      body.set("input_fingerprint", payload.input_fingerprint);
      await json(trigger.dataset.acceptUrl, post(body));
      if (token === generation) window.location.reload();
    } catch (error) {
      if (token === generation) status.textContent = error.message;
    } finally {
      if (token === generation) {
        busy = false;
        results.querySelectorAll("button").forEach(button => { button.disabled = false; });
      }
    }
  }
  async function generate(token) {
    if (busy || token !== generation) return;
    busy = true;
    actions.replaceChildren();
    status.textContent = "Queuing the observation's embedding…";
    try {
      const body = new FormData();
      body.set("scope", "observations");
      body.set("observation_ids", trigger.dataset.observationId);
      // Retain the key when retrying an uncertain HTTP response.
      if (!trigger.dataset.identityEmbeddingKey) trigger.dataset.identityEmbeddingKey = requestKey();
      body.set("request_key", trigger.dataset.identityEmbeddingKey);
      const run = await json(trigger.dataset.embeddingUrl, post(body));
      async function poll() {
        if (token !== generation) return;
        try {
          const state = await json(run.status_url);
          if (token !== generation) return;
          status.textContent = state.message || state.status_label;
          if (state.is_terminal || state.status === "attention") {
            busy = false;
            delete trigger.dataset.identityEmbeddingKey;
            if (state.status === "succeeded" || state.status === "partial") await load(token);
            else {
              status.textContent = state.message || "Embedding generation did not complete.";
              const link = element("a", "Open task details", "btn btn-outline-secondary");
              link.href = run.detail_url;
              actions.replaceChildren(link);
            }
          } else timer = setTimeout(poll, 1500);
        } catch (error) { if (token === generation) { busy = false; status.textContent = error.message; } }
      }
      await poll();
    } catch (error) {
      if (token === generation) { busy = false; status.textContent = error.message; }
    }
  }
  async function load(token) {
    const data = await json(trigger.dataset.candidatesUrl);
    if (token !== generation) return;
    payload = data;
    query.replaceChildren();
    const source = element("div");
    source.append(element("h3", "Selected observation", "fs-6"), preview(data.observation));
    query.append(source);
    if (!savedInputMatches(data.observation)) {
      status.textContent = "Save your annotation changes before requesting identity suggestions."; return;
    }
    if (data.status === "embedding_required") {
      status.textContent = "This observation needs an embedding for the selected model. Generate it to find candidates.";
      const button = element("button", "Generate embedding and find candidates", "btn btn-primary");
      button.type = "button";
      button.onclick = () => generate(token);
      actions.replaceChildren(button);
      return;
    }
    status.textContent = `Search complete. ${data.reference_count} of ${data.eligible_reference_count} eligible reference observations have usable embeddings. ` +
      "Cosine similarity is not a probability of identity.";
    if (!data.candidates.length) {
      results.append(element("p", "No matching reference identities are available. Generate embeddings for identified observations in Observations."));
    }
    data.candidates.forEach(candidate => {
      const column = element("div", "", "col-md-6");
      const card = element("div", "", "card h-100");
      const body = element("div", "", "card-body");
      const button = element("button", editorForm() ? "Use identity in form" : "Assign identity", "btn btn-primary");
      button.type = "button";
      button.onclick = () => accept(candidate, token);
      body.append(element("h3", candidate.label, "fs-5 mt-2"),
        element("p", `Similarity: ${candidate.score.toFixed(3)}`, "small"), button);
      card.append(preview(candidate.reference), body);
      column.append(card);
      results.append(column);
    });
  }
  document.addEventListener("click", async event => {
    const button = event.target.closest(".js-identity-candidates");
    if (!button) return;
    clearTimeout(timer);
    const token = ++generation;
    trigger = button;
    payload = null;
    busy = false;
    query.replaceChildren(); actions.replaceChildren(); results.replaceChildren();
    status.textContent = "Finding identity candidates…";
    bootstrap.Modal.getOrCreateInstance(modal).show();
    try { await load(token); }
    catch (error) { if (token === generation) status.textContent = error.message; }
  });
  modal.addEventListener("hidden.bs.modal", () => { ++generation; clearTimeout(timer); });
}());
