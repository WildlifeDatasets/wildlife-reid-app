// Exercise the real submit handler without browser/network dependencies.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const crypto = require("node:crypto").webcrypto;
const source = fs.readFileSync("src/api/caidapp/static/caidapp/embedding_runs.js", "utf8");

async function submit({ids = [], allFiltered = false, allowFiltered = true, filterQuery = "orientation=L&page=2"} = {}) {
  const label = {textContent: ""};
  const message = {textContent: "", hidden: true};
  const events = {};
  const form = {
    querySelectorAll: () => ids.map(value => ({value: String(value)})),
    querySelector: selector => selector === "[name=select_all_filtered]"
      ? {checked: allFiltered} : ids.length ? {value: String(ids[0])} : null,
    addEventListener() {},
  };
  const button = {
    dataset: {embeddingStart: "/caidapp/embedding-runs/", scope: "observations"},
    closest: () => form,
    querySelector: () => label,
    addEventListener: (name, callback) => { events[name] = callback; },
  };
  if (allowFiltered) Object.assign(button.dataset, {
    emptyScope: "filtered_observations", filteredScope: "filtered_observations", filterQuery,
  });
  let request = null;
  let redirect = null;
  vm.runInNewContext(source, {
    document: {
      querySelectorAll: () => [button],
      getElementById: () => message,
      querySelector: () => ({value: "csrf-token"}),
    },
    window: {crypto, location: {href: "http://localhost/caidapp/observations/?orientation=L", assign: url => { redirect = url; }}},
    crypto, FormData, URL, queueMicrotask,
    fetch: async (url, options) => {
      request = {url: new URL(url), body: options.body};
      return {ok: true, json: async () => ({detail_url: "/caidapp/tasks/example/"})};
    },
  });
  await events.click();
  return {request, redirect, label: label.textContent, message};
}

(async () => {
  let result = await submit();
  assert.equal(result.request.body.get("scope"), "filtered_observations");
  assert.deepEqual(result.request.body.getAll("observation_ids"), []);
  assert.equal(result.request.url.searchParams.get("orientation"), "L");
  assert.match(result.label, /all filtered observations/);
  assert.equal(result.redirect, "/caidapp/tasks/example/");

  result = await submit({ids: [17, 29]});
  assert.equal(result.request.body.get("scope"), "observations");
  assert.deepEqual(result.request.body.getAll("observation_ids"), ["17", "29"]);
  assert.match(result.label, /selected observations/);

  result = await submit({ids: [17], allFiltered: true});
  assert.equal(result.request.body.get("scope"), "filtered_observations");
  assert.deepEqual(result.request.body.getAll("observation_ids"), []);
  assert.equal(result.request.url.searchParams.get("orientation"), "L");
  assert.match(result.label, /all filtered observations/);

  result = await submit({filterQuery: ""});
  assert.equal(result.request.body.get("scope"), "filtered_observations");
  assert.equal(result.request.url.search, "");

  result = await submit({allowFiltered: false});
  assert.equal(result.request, null);
  assert.match(result.message.textContent, /Select at least one/);
  console.log("Passed 5 embedding selection flows: empty selection with filter, explicit, select-all-filtered, unfiltered, non-bulk guard.");
})().catch(error => { console.error(error); process.exitCode = 1; });
