/* Browser regression checks; run with Playwright available in NODE_PATH. */
const {chromium} = require("playwright");
const fs = require("node:fs");
const assert = require("node:assert/strict");

(async () => {
  const browser = await chromium.launch({headless: true, channel: "msedge"});
  try {
    const page = await browser.newPage({viewport: {width: 1280, height: 960}});
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    const item = {label: "Observation 1", image_url: "/image.svg", detail_url: "/media/1/", bbox: [0.5, 0.5, 0.5, 0.5]};
    const complete = {status: "complete", observation: item, expected_identity_id: null,
      input_fingerprint: "current-input", eligible_reference_count: 4, reference_count: 3,
      candidates: [{identity_id: 7, label: "Lynx <A>", score: 0.95, reference: item}]};
    let mode = "complete";
    let generated = false;
    let accepted = 0;
    let starts = 0;
    await page.route("http://identity.test/**", async route => {
      const url = new URL(route.request().url());
      if (url.pathname === "/candidates") return route.fulfill({json:
        mode === "missing" && !generated ? {...complete, status: "embedding_required", candidates: []} : complete});
      if (url.pathname === "/accept") {
        accepted++;
        const body = route.request().postData();
        assert.ok(body.includes("current-input"));
        assert.ok(body.includes("7"));
        return route.fulfill({json: {identity_id: 7}});
      }
      if (url.pathname === "/generate") {
        starts++;
        assert.ok(route.request().postData().includes("observation_ids"));
        return route.fulfill({json: {status_url: "/status", detail_url: "/task"}});
      }
      if (url.pathname === "/status") {
        generated = true;
        return route.fulfill({json: {status: "succeeded", is_terminal: true}});
      }
      if (url.pathname === "/image.svg") return route.fulfill({contentType: "image/svg+xml",
        body: '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="400"><rect width="600" height="400" fill="#8da98b"/><circle cx="300" cy="200" r="90" fill="#c2a475"/></svg>'});
      return route.fulfill({body: "<html><body></body></html>"});
    });
    const button = '<button type="button" class="js-identity-candidates btn btn-primary" data-candidates-url="/candidates" data-accept-url="/accept" data-embedding-url="/generate" data-observation-id="1">Suggest identity</button>';
    async function setup(editor = false) {
      await page.goto("http://identity.test/");
      await page.setContent('<input type="hidden" name="csrfmiddlewaretoken" value="test-token">' +
        (editor ? '<div class="observation-form"><input name="observations-0-bbox_x_center" value="0.5">' +
          '<input name="observations-0-bbox_y_center" value="0.5"><input name="observations-0-bbox_width" value="0.5">' +
          '<input name="observations-0-bbox_height" value="0.5"><select name="observations-0-identity"><option value=""></option></select>' +
          button.replace('data-observation-id="1"', 'data-observation-id="1" data-editor="true"') + '</div>' : button) +
        fs.readFileSync("src/api/caidapp/templates/caidapp/includes/identity_candidates_modal.html", "utf8"));
      await page.addStyleTag({path: "node_modules/bootstrap/dist/css/bootstrap.min.css"});
      await page.addScriptTag({path: "node_modules/bootstrap/dist/js/bootstrap.bundle.min.js"});
      await page.addScriptTag({path: "src/api/caidapp/static/caidapp/js/identity-candidates.js"});
      await page.getByRole("button", {name: "Suggest identity", exact: true}).click();
    }
    await setup();
    await page.getByRole("heading", {name: "Lynx <A>"}).waitFor();
    await page.waitForFunction(() => getComputedStyle(document.querySelector(".modal-dialog")).transform === "none");
    assert.match(await page.locator("[data-candidate-status]").textContent(), /3 of 4/);
    assert.equal(await page.locator("[data-candidate-results] canvas").count(), 1);
    await page.screenshot({path: ".codex_tmp/identity-candidates-desktop.png"});
    await page.setViewportSize({width: 390, height: 844});
    await page.screenshot({path: ".codex_tmp/identity-candidates-mobile.png"});
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
    await Promise.all([
      page.waitForEvent("load"),
      page.getByRole("button", {name: "Assign identity", exact: true}).click()
    ]);
    assert.equal(accepted, 1);

    mode = "missing";
    await setup();
    await page.getByRole("button", {name: "Generate embedding and find candidates"}).click();
    await page.getByRole("heading", {name: "Lynx <A>"}).waitFor();
    assert.equal(starts, 1);

    mode = "complete";
    await setup(true);
    await page.getByRole("button", {name: "Use identity in form"}).click();
    assert.equal(await page.locator("select[name$='-identity']").inputValue(), "7");
    assert.match(await page.locator("[data-candidate-status]").textContent(), /Save the media file/);
    assert.equal(accepted, 1);

    await setup(true);
    await page.getByRole("button", {name: "Use identity in form"}).waitFor();
    await page.locator("input[name$='-bbox_width']").fill("0.3");
    await page.getByRole("button", {name: "Use identity in form"}).click();
    assert.match(await page.locator("[data-candidate-status]").textContent(), /Save your annotation/);
    assert.equal(await page.locator("select[name$='-identity']").inputValue(), "");
    assert.deepEqual(errors, []);
    console.log("PASS: candidates, safe labels, desktop/mobile, assignment, embedding generation, editor selection and stale crop guard");
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
