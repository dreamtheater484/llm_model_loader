import assert from "node:assert/strict";
import { setImmediate } from "node:timers/promises";
import test from "node:test";
import { createDashboardRefresh } from "../src/refresh.js";

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function setup(t, request) {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const state = {};
  const setters = Object.fromEntries(
    ["settings", "telemetry", "models", "downloads", "runs", "presets"]
      .map((key) => [key, (data) => { state[key] = data; }])
  );
  return { state, ...createDashboardRefresh(request, setters) };
}

async function flush(t) {
  t.mock.timers.tick(100);
  await setImmediate();
}

test("loaded and aborted states render while telemetry is blocked or failing", async (t) => {
  const telemetry = deferred();
  let status = "loaded";
  const app = setup(t, async (path) => {
    if (path === "/api/system/telemetry") return telemetry.promise;
    return path === "/api/runs" ? [{ status }] : [];
  });
  const reload = app.reload();
  const failed = assert.rejects(reload, /telemetry unavailable/);
  await flush(t);
  assert.equal(app.state.runs[0].status, "loaded");
  assert.equal(app.state.telemetry, undefined);
  status = "aborted";
  const event = app.onEvent({ type: "run", payload: { status } });
  await flush(t);
  await event;
  assert.equal(app.state.runs[0].status, "aborted");
  telemetry.reject(new Error("telemetry unavailable"));
  await failed;
  assert.equal(app.state.runs[0].status, "aborted");
});

test("a thousand log events request one run snapshot and no hardware probes", async (t) => {
  const paths = [];
  const app = setup(t, async (path) => { paths.push(path); return []; });
  const events = Array.from({ length: 1000 }, () => app.onEvent({ type: "run_log" }));
  await flush(t);
  await Promise.all(events);
  assert.deepEqual(paths, ["/api/runs"]);
});

test("events during an in-flight read coalesce into a fresh serialized read", async (t) => {
  const first = deferred();
  let reads = 0;
  const app = setup(t, () => ++reads === 1 ? first.promise : [{ status: "unloaded" }]);
  const loading = app.onEvent({ type: "run" });
  await flush(t);
  const events = Array.from({ length: 1000 }, () => app.onEvent({ type: "run_log" }));
  await flush(t);
  assert.equal(reads, 1);
  first.resolve([{ status: "loading" }]);
  await setImmediate();
  await flush(t);
  await Promise.all([loading, ...events]);
  assert.equal(reads, 2);
  assert.equal(app.state.runs[0].status, "unloaded");
});

test("a failed run refresh can be retried", async (t) => {
  let reads = 0;
  const app = setup(t, async () => {
    if (++reads === 1) throw new Error("offline");
    return [{ status: "loaded" }];
  });
  const failed = assert.rejects(app.onEvent({ type: "run" }), /offline/);
  await flush(t);
  await failed;
  const retry = app.onEvent({ type: "run" });
  await flush(t);
  await retry;
  assert.equal(app.state.runs[0].status, "loaded");
});

test("downloads refresh their queue and register completed models without telemetry", async (t) => {
  const paths = [];
  const app = setup(t, async (path) => { paths.push(path); return []; });
  const progress = app.onEvent({ type: "download", payload: { status: "downloading" } });
  await flush(t);
  await progress;
  assert.deepEqual(paths.splice(0), ["/api/downloads"]);
  const completed = app.onEvent({ type: "download", payload: { status: "completed" } });
  await flush(t);
  await completed;
  assert.deepEqual(paths.splice(0).sort(), ["/api/downloads", "/api/models"]);
  await app.onEvent({ type: "benchmark" });
  assert.deepEqual(paths, []);
});
