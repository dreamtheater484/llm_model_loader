// Batch bursts and serialize reads of each resource. An event arriving during a
// read requests one follow-up so an older response cannot hide a lifecycle change.
function resourceRefresh(request, path, apply) {
  let pending = null;
  let dirty = false;
  return () => {
    dirty = true;
    if (!pending) {
      pending = (async () => {
        try {
          do {
            await new Promise((resolve) => setTimeout(resolve, 100));
            dirty = false;
            apply(await request(path));
          } while (dirty);
        } finally {
          pending = null;
        }
      })();
    }
    return pending;
  };
}

export function createDashboardRefresh(request, setters) {
  const paths = {
    settings: "/api/settings",
    telemetry: "/api/system/telemetry",
    models: "/api/models",
    downloads: "/api/downloads",
    runs: "/api/runs",
    presets: "/api/benchmarks"
  };
  const refresh = Object.fromEntries(Object.entries(paths).map(([key, path]) => [
    key, resourceRefresh(request, path, setters[key])
  ]));
  return {
    // Each response updates its own panel immediately, even if another is slow
    // or fails. Waiting for all results here only aggregates refresh errors.
    async reload() {
      const results = await Promise.allSettled(Object.values(refresh).map((load) => load()));
      const failure = results.find((result) => result.status === "rejected");
      if (failure) throw failure.reason;
    },
    async onEvent({ type, payload }) {
      if (type === "run" || type === "run_log") {
        await refresh.runs();
      } else if (type === "download") {
        await Promise.all([
          refresh.downloads(),
          ...(payload?.status === "completed" ? [refresh.models()] : [])
        ]);
      }
      // Benchmark history has its own polling; it does not invalidate these panels.
    }
  };
}
