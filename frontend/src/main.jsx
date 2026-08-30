import React, { useCallback, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  DndContext,
  DragOverlay,
  KeyboardSensor,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors
} from "@dnd-kit/core";
import {
  SortableContext,
  arrayMove,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import {
  Activity,
  BarChart3,
  Boxes,
  ChevronUp,
  CircleDollarSign,
  Cpu,
  Download,
  ChevronDown,
  ChevronRight,
  FileArchive,
  Folder,
  FolderOpen,
  Gauge,
  GripVertical,
  HardDrive,
  Home,
  MemoryStick,
  Pencil,
  Play,
  Plus,
  Power,
  RefreshCw,
  Save,
  Search,
  Square,
  Star,
  Timer,
  Trash2,
  Upload,
  X,
  Zap
} from "lucide-react";
import "./styles.css";

const API = "";

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) }
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || response.statusText);
  }
  return response.json();
}

function formatBytes(value) {
  if (!value && value !== 0) return "unknown";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(index ? 1 : 0)} ${units[index]}`;
}

function formatMib(value) {
  if (!value && value !== 0) return "unknown";
  return value >= 1024 ? `${(value / 1024).toFixed(1)} GB` : `${Math.round(value)} MiB`;
}

function formatDuration(seconds) {
  if (!seconds && seconds !== 0) return "0s";
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m ${String(s).padStart(2, "0")}s`;
  if (m) return `${m}m ${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

function formatTokens(value) {
  const tokens = Number(value || 0);
  if (!Number.isFinite(tokens)) return "0";
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(tokens >= 10_000_000 ? 0 : 1)}M`;
  if (tokens >= 1_000) return `${(tokens / 1_000).toFixed(tokens >= 10_000 ? 0 : 1)}k`;
  return tokens.toLocaleString();
}

function formatTokenRate(value) {
  if (value == null) return "—";
  const rate = Number(value);
  if (!Number.isFinite(rate)) return "—";
  if (rate >= 1000) return `${(rate / 1000).toFixed(rate >= 10_000 ? 0 : 1)}k`;
  return rate.toFixed(rate >= 100 ? 0 : 1);
}

function formatPower(value) {
  const watts = Number(value);
  if (!Number.isFinite(watts)) return "—";
  return `${watts.toFixed(watts >= 100 ? 0 : 1)} W`;
}

function formatEnergy(value, historical = false) {
  const kwh = Number(value);
  if (!Number.isFinite(kwh)) return "—";
  if (historical && kwh >= 1000) return `${(kwh / 1000).toFixed(kwh >= 10_000 ? 1 : 2)} MWh`;
  const digits = kwh < 0.1 ? 4 : kwh < 10 ? 3 : kwh < 100 ? 2 : 1;
  return `${kwh.toFixed(digits)} kWh`;
}

function formatCost(value, currency = "USD") {
  if (value == null || value === "") return "—";
  const amount = Number(value);
  if (!Number.isFinite(amount)) return "—";
  try {
    return new Intl.NumberFormat(undefined, { style: "currency", currency, minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(amount);
  } catch {
    return `${currency} ${amount.toFixed(2)}`;
  }
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString();
}

function percent(done, total) {
  if (!total) return 0;
  return Math.max(0, Math.min(100, (done / total) * 100));
}

function compactLogTail(text) {
  if (!text) return "";
  const lines = text.split(/\r?\n/);
  return lines.filter((line, index) => index === 0 || line !== lines[index - 1]).join("\n");
}

function Progress({ value }) {
  return (
    <div className="progress" aria-label="progress">
      <span style={{ width: `${Math.max(2, value)}%` }} />
    </div>
  );
}

function Pill({ children, tone = "neutral", title }) {
  return <span className={`pill ${tone}`} title={title}>{children}</span>;
}

function IconButton({ icon: Icon, label, className = "", ...props }) {
  return (
    <button className={`iconButton ${className}`.trim()} title={label} aria-label={label} {...props}>
      <Icon size={15} />
    </button>
  );
}

function TopTelemetry({ telemetry, usage, refresh }) {
  const gpu = telemetry?.gpus?.[0];
  const used = gpu ? percent(gpu.memory_used_mib, gpu.memory_total_mib) : 0;
  const memoryUsed = telemetry?.memory_used_bytes;
  const memoryTotal = telemetry?.memory_total_bytes;
  const memoryPercent = percent(memoryUsed, memoryTotal);
  const memoryDetails = [
    telemetry?.memory_type || "RAM",
    telemetry?.memory_speed_mts ? `${telemetry.memory_speed_mts} MT/s` : null
  ].filter(Boolean).join(" · ");
  const sessionUsage = usage?.recent_task?.usage;
  const sessionCosts = sessionUsage?.costs || {};
  const currencies = Array.from(new Set((usage?.models || []).map((item) => item.price_currency || "USD")));
  const sessionCurrency = currencies.length === 1 ? currencies[0] : "USD";
  const mixedCurrencies = currencies.length > 1;
  const sessionCost = (value, category) => {
    if (!sessionUsage || sessionUsage.cost_status === "unpriced" || sessionUsage.missing_rates?.includes(category)) return "—";
    return `${sessionUsage.cost_status === "partially_priced" ? "~" : ""}${mixedCurrencies ? value || "0" : formatCost(value, sessionCurrency)}`;
  };
  const sessionTotal = sessionUsage?.cost_status === "unpriced"
    ? "Unpriced"
    : sessionUsage
      ? `${sessionUsage.cost_status === "partially_priced" ? "~" : ""}${mixedCurrencies ? sessionUsage.cost || "0" : formatCost(sessionUsage.cost, sessionCurrency)}`
      : "—";
  const cacheCategoryMissing = sessionUsage?.missing_rates?.includes("cache_read") || sessionUsage?.missing_rates?.includes("cache_write");
  const tokenSpeed = telemetry?.token_speed;
  const power = telemetry?.power;
  const speedLine = (scope) => `Prep ${formatTokenRate(tokenSpeed?.preprocessing?.[scope])} · Decode ${formatTokenRate(tokenSpeed?.decode?.[scope])}`;
  return (
    <header className="topbar">
      <div className="brand">
        <span className="brandMark"><Boxes size={19} /></span>
        <div>
          <strong>LLM Model Loader</strong>
          <span>llama.cpp control surface</span>
        </div>
      </div>
      <div className="telemetryGroups">
        <section className="telemetryGroup systemTelemetry">
          <div className="telemetryGroupLabel"><i />System</div>
          <div className="systemMetrics">
            <div className="dashboardMetric memoryMetric">
              <span className="metricIcon memoryGpu"><HardDrive size={15} /></span>
              <div>
                <label>{gpu?.name || "No NVIDIA GPU detected"}</label>
                <b>{gpu ? `${formatMib(gpu.memory_free_mib)} free / ${formatMib(gpu.memory_total_mib)}` : "Unavailable"}</b>
                <Progress value={used} />
              </div>
            </div>
            <div className="dashboardMetric memoryMetric">
              <span className="metricIcon memoryRam"><MemoryStick size={15} /></span>
              <div>
                <label>{memoryDetails}</label>
                <b>{memoryTotal == null ? "Unavailable" : `${formatBytes(memoryUsed)} used / ${formatBytes(memoryTotal)}`}</b>
                <Progress value={memoryPercent} />
              </div>
            </div>
            <div className="dashboardMetric compactMetric">
              <span className="metricIcon computeUtil"><Gauge size={15} /></span>
              <div><label>Util</label><b>{gpu?.utilization_gpu_percent ?? 0}%</b></div>
            </div>
            <div className="dashboardMetric compactMetric">
              <span className="metricIcon computeCpu"><Cpu size={15} /></span>
              <div><label>CPU</label><b>{telemetry?.cpu_load_percent == null ? "n/a" : `${telemetry.cpu_load_percent.toFixed(0)}%`}</b></div>
            </div>
          </div>
        </section>
        <section className="telemetryGroup powerTelemetry">
          <div className="telemetryGroupLabel powerGroupLabel"><i />Power</div>
          <button className="powerSummary" type="button" aria-describedby="power-breakdown">
            <span className="metricIcon powerIcon"><Zap size={15} /></span>
            <span className="powerStat powerCurrent">
              <label>Current</label>
              <b>{formatPower(power?.current_system_w)}</b>
            </span>
            <span className="powerStat">
              <label>Session</label>
              <b>{formatEnergy(power?.session_kwh)}</b>
            </span>
            <span className="powerStat">
              <label>All time</label>
              <b>{formatEnergy(power?.total_ever_kwh, true)}</b>
            </span>
          </button>
          <div className="powerPopover" id="power-breakdown" role="tooltip">
            <strong>Estimated system power</strong>
            <span><em>GPU board · measured</em><b>{formatPower(power?.gpu_w)}</b></span>
            <span><em>CPU package · measured</em><b>{formatPower(power?.cpu_w)}</b></span>
            <span><em>Platform · estimated</em><b>{formatPower(power?.platform_overhead_w)}</b></span>
            <span><em>PSU efficiency</em><b>{power?.psu_efficiency ? `${Math.round(power.psu_efficiency * 100)}%` : "—"}</b></span>
            <span className="powerPopoverTotal"><em>Current wall estimate</em><b>{formatPower(power?.current_system_w)}</b></span>
            <small>Session starts with the backend. All-time energy is recorded from when power tracking was enabled.</small>
          </div>
        </section>
        <section className="telemetryGroup sessionTelemetry">
          <div className="telemetryGroupLabel sessionGroupLabel"><i />Runtime &amp; Session</div>
          <div className="sessionMetrics">
            <div className="modelMetric">
              <label>Models</label>
              <b><i />{telemetry?.loaded_models || 0} live / {telemetry?.loading_models || 0} loading</b>
            </div>
            <div className="dashboardMetric sessionCostMetric" title={usage?.recent_task?.title || "Most recently updated OpenCode task"}>
              <span className="metricIcon sessionCostIcon"><CircleDollarSign size={15} /></span>
              <div>
                <label>Session cost</label>
                <b>{sessionTotal}</b>
                <small>{sessionUsage ? `in ${sessionCost(sessionCosts.input, "input")} · out ${sessionCost(sessionCosts.output, "output")} · cache ${cacheCategoryMissing ? "—" : `${sessionUsage.cost_status === "partially_priced" ? "~" : ""}${mixedCurrencies ? sessionCosts.cache || "0" : formatCost(sessionCosts.cache, sessionCurrency)}`}` : "No OpenCode session"}</small>
              </div>
            </div>
            <div
              className="dashboardMetric speedMetric"
              title="Current is the latest runtime sample. Session average is weighted from server start. Preprocessing excludes cached prompt tokens."
            >
              <span className="metricIcon speedIcon"><Timer size={15} /></span>
              <div>
                <label>Token speed · tok/s</label>
                <b><span>Current</span>{speedLine("current_tps")}</b>
                <small><span>Session avg</span>{speedLine("session_average_tps")}</small>
              </div>
            </div>
          </div>
        </section>
      </div>
      <IconButton icon={RefreshCw} label="Refresh telemetry" onClick={refresh} />
    </header>
  );
}

function Settings({ settings, setSettings, toast }) {
  const [draft, setDraft] = useState(settings);
  const [browser, setBrowser] = useState(null);
  const [openBrowser, setOpenBrowser] = useState(false);
  const [message, setMessage] = useState("");
  useEffect(() => setDraft(settings), [settings]);

  async function save() {
    const updated = await request("/api/settings", { method: "PATCH", body: JSON.stringify(draft) });
    setSettings(updated);
    toast("Settings saved");
  }

  async function discover() {
    try {
      const discovered = await request("/api/llamacpp/discover");
      if (discovered.selected) {
        const updated = { ...draft, llama_server_path: discovered.selected };
        setDraft(updated);
        setSettings({ ...settings, llama_server_path: discovered.selected });
        setMessage(`Found ${discovered.selected}`);
        toast("llama.cpp path discovered");
      } else {
        setMessage("No llama-server.exe found.");
      }
    } catch (error) {
      setMessage(error.message);
      toast(error.message);
    }
  }

  async function loadBrowser(nextPath) {
    try {
      const query = nextPath ? `?path=${encodeURIComponent(nextPath)}&gguf_only=false&executable_only=true` : "?gguf_only=false&executable_only=true";
      setBrowser(await request(`/api/files/browse${query}`));
      setOpenBrowser(true);
    } catch (error) {
      setMessage(error.message);
      toast(error.message);
    }
  }

  function selectExecutable(path) {
    setDraft({ ...draft, llama_server_path: path });
    setOpenBrowser(false);
    setMessage(`Selected ${path}`);
  }

  return (
    <section className="band settingsPanel">
      <div className="settingsBand">
        <div className="sectionTitle">
          <Power size={16} />
          <h2>Runtime Settings</h2>
        </div>
        <input value={draft.llama_server_path || ""} onChange={(event) => setDraft({ ...draft, llama_server_path: event.target.value })} placeholder="Path to llama-server.exe" />
        <input value={draft.model_dir || ""} onChange={(event) => setDraft({ ...draft, model_dir: event.target.value })} placeholder="Managed model directory" />
        <input value={draft.opencode_db_path || ""} onChange={(event) => setDraft({ ...draft, opencode_db_path: event.target.value })} placeholder="OpenCode DB path (optional)" />
        <button onClick={discover}><Search size={14} /> Auto</button>
        <button onClick={() => loadBrowser(draft.llama_server_path ? draft.llama_server_path.replace(/\\[^\\]*$/, "") : "")}><FolderOpen size={14} /> Browse</button>
        <button className="primary" onClick={save}><Save size={14} /> Save</button>
      </div>
      {message && <div className="settingsMessage">{message}</div>}
      {openBrowser && (
        <div className="browserShell runtimeBrowser">
          <div className="browserChrome">
            <button onClick={() => loadBrowser(browser?.parent)} disabled={!browser?.parent}><ChevronUp size={14} /> Up</button>
            <button onClick={() => loadBrowser("D:\\Documents\\AI")}><Home size={14} /> AI</button>
            <input value={browser?.path || ""} onChange={(event) => setBrowser({ ...(browser || {}), path: event.target.value })} onKeyDown={(event) => event.key === "Enter" && loadBrowser(event.currentTarget.value)} />
            <button onClick={() => loadBrowser(browser?.path)}><FolderOpen size={14} /> Open</button>
          </div>
          <div className="shortcutStrip">
            {(browser?.roots || []).map((root) => <button key={root.path} onClick={() => loadBrowser(root.path)}>{root.name}</button>)}
            {(browser?.shortcuts || []).map((shortcut) => <button key={shortcut.path} onClick={() => loadBrowser(shortcut.path)}>{shortcut.name}</button>)}
          </div>
          <div className="fileBrowser compact">
            {!browser?.entries?.length && <div className="empty">No llama-server.exe in this folder.</div>}
            {browser?.entries?.map((entry) => (
              <button
                className="fileEntry"
                key={entry.path}
                onClick={() => entry.type === "directory" ? loadBrowser(entry.path) : selectExecutable(entry.path)}
              >
                {entry.type === "directory" ? <Folder size={15} /> : <Power size={15} />}
                <span>{entry.name}</span>
                {entry.type === "file" && <small>llama-server</small>}
              </button>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function Discover({ toast, reload, telemetry }) {
  const [q, setQ] = useState("Qwen GGUF");
  const [results, setResults] = useState([]);
  const [files, setFiles] = useState({});
  const [expanded, setExpanded] = useState({});
  const [loadingFiles, setLoadingFiles] = useState({});
  const [loading, setLoading] = useState(false);
  const [resultsCollapsed, setResultsCollapsed] = useState(true);

  async function search() {
    setLoading(true);
    try {
      setResults(await request(`/api/hf/search?q=${encodeURIComponent(q)}&limit=20`));
      setResultsCollapsed(false);
    } catch (error) {
      toast(error.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadFiles(repo) {
    if (expanded[repo]) {
      setExpanded({ ...expanded, [repo]: false });
      return;
    }
    setExpanded({ ...expanded, [repo]: true });
    if (files[repo]) return;
    setLoadingFiles({ ...loadingFiles, [repo]: true });
    try {
      setFiles({ ...files, [repo]: await request(`/api/hf/models/${repo}/files`) });
    } catch (error) {
      toast(error.message);
    } finally {
      setLoadingFiles((current) => ({ ...current, [repo]: false }));
    }
  }

  async function download(repo, file) {
    try {
      await request("/api/downloads", {
        method: "POST",
        body: JSON.stringify({ repo_id: repo, filename: file.filename, filenames: file.filenames })
      });
      toast(file.shard_count > 1 ? `All ${file.shard_count} parts queued` : "Download queued");
      reload();
    } catch (error) {
      toast(error.message);
    }
  }

  function fitLabel(file) {
    const freeMib = telemetry?.gpus?.[0]?.memory_free_mib;
    if (!file.estimated_vram_mib || !freeMib) return "VRAM estimate unknown";
    return file.estimated_vram_mib + 1024 <= freeMib ? "fits current VRAM" : "exceeds free VRAM";
  }

  function fitTone(file) {
    const freeMib = telemetry?.gpus?.[0]?.memory_free_mib;
    if (!file.estimated_vram_mib || !freeMib) return "neutral";
    return file.estimated_vram_mib + 1024 <= freeMib ? "completed" : "failed";
  }

  return (
    <section className="band discoveryBand">
      <div className="sectionTitle">
        <Search size={16} />
        <h2>Hugging Face Discovery</h2>
      </div>
      <div className="toolbar">
        <input value={q} onChange={(event) => setQ(event.target.value)} onKeyDown={(event) => event.key === "Enter" && search()} placeholder="Search GGUF models" />
        <button className="primary" onClick={search}><Search size={14} /> Search</button>
        {!!results.length && (
          <button className="collapseButton" onClick={() => setResultsCollapsed(!resultsCollapsed)}>
            {resultsCollapsed ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
            {resultsCollapsed ? "Show results" : "Hide results"}
          </button>
        )}
      </div>
      {!resultsCollapsed && <div className="table">
        <div className="thead repoGrid"><span>Repository</span><span>Stats</span><span>Tags</span><span>Action</span></div>
        {loading && <div className="empty">Searching Hugging Face...</div>}
        {results.map((result) => (
          <React.Fragment key={result.repo_id}>
            <div className="row repoGrid">
              <strong>{result.repo_id}</strong>
              <span>{result.downloads} downloads / {result.likes} likes</span>
              <span className="tagLine">{(result.tags || []).slice(0, 4).map((tag) => <Pill key={tag}>{tag}</Pill>)}</span>
              <button className="primary openRepoButton" onClick={() => loadFiles(result.repo_id)}>
                {expanded[result.repo_id] ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                {expanded[result.repo_id] ? "Close" : "Open"}
              </button>
            </div>
            {expanded[result.repo_id] && (
              <div className="variantPanel">
                <div className="variantHead fileGrid">
                  <span>Version</span>
                  <span>Quant</span>
                  <span>Size</span>
                  <span>VRAM</span>
                  <span></span>
                </div>
                {loadingFiles[result.repo_id] && <div className="empty">Loading GGUF variants...</div>}
                {!loadingFiles[result.repo_id] && !files[result.repo_id]?.length && <div className="empty">No GGUF files found for this repository.</div>}
                {files[result.repo_id]?.map((file) => (
                  <div className="row fileGrid" key={`${result.repo_id}-${file.filename}`}>
                    <span>
                      {file.display_name || file.filename}
                      {file.shard_count > 1 && <small>{file.shard_count} parts</small>}
                    </span>
                    <span>{file.quantization || "unknown"}</span>
                    <strong>{formatBytes(file.size_bytes)}</strong>
                    <span className="vramEstimate">
                      {file.estimated_vram_mib ? formatMib(file.estimated_vram_mib) : "unknown"}
                      <Pill tone={fitTone(file)}>{fitLabel(file)}</Pill>
                    </span>
                    <button className="primary" disabled={!file.complete} onClick={() => download(result.repo_id, file)}>
                      <Download size={14} /> {file.shard_count > 1 ? "Download all" : "Download"}
                    </button>
                  </div>
                ))}
              </div>
            )}
          </React.Fragment>
        ))}
      </div>}
    </section>
  );
}

function Downloads({ downloads, reload, toast }) {
  const [open, setOpen] = useState(false);
  const [page, setPage] = useState(0);
  const pageSize = 7;
  const activeStatuses = new Set(["queued", "running", "retrying"]);
  const activeDownloads = downloads.filter((download) => activeStatuses.has(download.status));
  const visibleDownloads = open ? downloads.slice(page * pageSize, page * pageSize + pageSize) : activeDownloads;
  const pages = Math.max(1, Math.ceil(downloads.length / pageSize));
  async function cancel(download) {
    try {
      await request(`/api/downloads/${download.id}/cancel`, { method: "POST" });
      reload();
      toast("Download cancellation requested");
    } catch (error) {
      toast(error.message);
    }
  }
  async function resume(download) {
    try {
      await request(`/api/downloads/${download.id}/resume`, { method: "POST" });
      reload();
      toast("Download resumed");
    } catch (error) {
      toast(error.message);
    }
  }
  return (
    <section className="band">
      <div className="sectionTitle">
        <Download size={16} />
        <h2>Download Queue</h2>
        <div className="spacer" />
        <button className="subtle" onClick={() => setOpen(!open)}>
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          {open ? "Hide history" : `Open history (${downloads.length})`}
        </button>
      </div>
      <div className="table">
        <div className="thead downloadGrid"><span>File</span><span>Status</span><span>Progress</span><span>Transfer</span><span></span></div>
        {!visibleDownloads.length && <div className="empty">{open ? "No downloads yet." : "No active downloads."}</div>}
        {visibleDownloads.map((download) => {
          const p = percent(download.bytes_done, download.bytes_total);
          return (
            <div className="row downloadGrid" key={download.id}>
              <strong>{download.filename}</strong>
              <Pill tone={download.status}>{download.status}</Pill>
              <div><Progress value={p} /><small>{p.toFixed(1)}% done / {(100 - p).toFixed(1)}% left</small></div>
              <span>{formatBytes(download.bytes_done)} / {formatBytes(download.bytes_total)}</span>
              <span className="downloadActions">
                {["queued", "running", "retrying"].includes(download.status) && <button onClick={() => cancel(download)}><Square size={14} /> Cancel</button>}
                {["paused", "failed", "cancelled"].includes(download.status) && <button onClick={() => resume(download)}><RefreshCw size={14} /> Resume</button>}
              </span>
            </div>
          );
        })}
      </div>
      {open && downloads.length > pageSize && (
        <div className="pager">
          <button onClick={() => setPage(Math.max(0, page - 1))} disabled={page === 0}>Previous</button>
          <span>Page {page + 1} / {pages}</span>
          <button onClick={() => setPage(Math.min(pages - 1, page + 1))} disabled={page >= pages - 1}>Next</button>
        </div>
      )}
    </section>
  );
}

function ImportModel({ reload, toast, settings }) {
  const [path, setPath] = useState("");
  const [name, setName] = useState("");
  const [manual, setManual] = useState("");
  const [browser, setBrowser] = useState(null);
  const [copyToManaged, setCopyToManaged] = useState(false);
  const [importing, setImporting] = useState(false);
  const [importMessage, setImportMessage] = useState("");

  const loadBrowser = useCallback(async (nextPath) => {
    try {
      setImportMessage("");
      const query = nextPath ? `?path=${encodeURIComponent(nextPath)}` : "";
      const data = await request(`/api/files/browse${query}`);
      setBrowser(data);
    } catch (error) {
      setImportMessage(error.message);
      toast(error.message);
    }
  }, [toast]);

  useEffect(() => {
    loadBrowser(settings?.model_dir || "");
  }, [loadBrowser, settings?.model_dir]);

  async function submit() {
    await importSelected(path);
  }

  async function importSelected(selectedPath) {
    if (!selectedPath) {
      setImportMessage("Select a model file first.");
      return;
    }
    if (!path) {
      setPath(selectedPath);
    }
    setImporting(true);
    setImportMessage(copyToManaged ? "Copying model into the managed library..." : "Registering selected model...");
    try {
      const imported = await request("/api/models", {
        method: "POST",
        body: JSON.stringify({ path: selectedPath, name: name || null, manual_vram_mib: manual ? Number(manual) : null, copy_to_managed: copyToManaged })
      });
      setPath("");
      setName("");
      setManual("");
      const message = imported.already_exists ? "That model is already in the library." : "Model imported.";
      setImportMessage(message);
      reload();
      toast(message);
    } catch (error) {
      setImportMessage(error.message);
      toast(error.message);
    } finally {
      setImporting(false);
    }
  }

  return (
    <section className="band">
      <div className="sectionTitle">
        <Upload size={16} />
        <h2>Import Existing Model</h2>
      </div>
      <div className="browserShell">
        <div className="browserChrome">
          <button onClick={() => loadBrowser(browser?.parent)} disabled={!browser?.parent}><ChevronUp size={14} /> Up</button>
          <button onClick={() => loadBrowser(settings?.model_dir)}><Home size={14} /> Models</button>
          <input value={browser?.path || ""} onChange={(event) => setBrowser({ ...(browser || {}), path: event.target.value })} onKeyDown={(event) => event.key === "Enter" && loadBrowser(event.currentTarget.value)} />
          <button onClick={() => loadBrowser(browser?.path)}><FolderOpen size={14} /> Open</button>
        </div>
        <div className="shortcutStrip">
          {(browser?.roots || []).map((root) => (
            <button key={root.path} onClick={() => loadBrowser(root.path)}>{root.name}</button>
          ))}
          {(browser?.shortcuts || []).map((shortcut) => (
            <button key={shortcut.path} onClick={() => loadBrowser(shortcut.path)}>{shortcut.name}</button>
          ))}
        </div>
        {browser?.error && <div className="inlineError">{browser.error}</div>}
        <div className="fileBrowser">
          {!browser?.entries?.length && <div className="empty">No GGUF or NInfer files in this folder.</div>}
          {browser?.entries?.map((entry) => (
            <button
              className={`fileEntry ${entry.path === path ? "selected" : ""} ${entry.format === "ninfer" ? "ninfer" : ""}`}
              key={entry.path}
              onClick={() => entry.type === "directory" ? loadBrowser(entry.path) : (setPath(entry.path), setImportMessage(""))}
              onDoubleClick={() => entry.type === "directory" ? loadBrowser(entry.path) : importSelected(entry.path)}
            >
              {entry.type === "directory" ? <Folder size={15} /> : <FileArchive size={15} />}
              <span>{entry.name}</span>
              {entry.type === "file" && (
                <span className="fileMeta">
                  <small>{entry.format === "ninfer" ? formatBytes(entry.size_bytes) : `${entry.quantization || "GGUF"} / ${formatBytes(entry.size_bytes)}`}</small>
                  {entry.format === "ninfer" && <Pill tone="ninfer">NInfer</Pill>}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>
      <div className={`selectedFile ${path ? "ready" : ""}`}>
        <FileArchive size={14} />
        <span>{path || "No model file selected. Open folders above, then select a .gguf or .ninfer row."}</span>
      </div>
      {importMessage && <div className={`importMessage ${importMessage.includes("imported") ? "ok" : ""}`}>{importMessage}</div>}
      <div className="toolbar wrap">
        <input value={path} onChange={(event) => setPath(event.target.value)} placeholder="Selected .gguf file" />
        <input value={name} onChange={(event) => setName(event.target.value)} placeholder="Optional display name" />
        <input value={manual} onChange={(event) => setManual(event.target.value)} placeholder="Manual VRAM MiB" />
        <label className="checkLine"><input type="checkbox" checked={copyToManaged} onChange={(event) => setCopyToManaged(event.target.checked)} /> Copy into managed library (slower)</label>
        <button className="primary" onClick={submit} disabled={!path || importing}><Plus size={14} /> {importing ? "Importing" : "Import"}</button>
      </div>
    </section>
  );
}

function ScriptEditor({ model, reload, toast }) {
  const template = `-m "${model.path}" \`
  --alias "${model.name}" \`
  --host 127.0.0.1 \`
  --port 8080 \`
  -c 32768 \`
  -ngl 99 \`
  -fa on \`
  --log-verbosity 1`;
  const [raw, setRaw] = useState(template);
  const [name, setName] = useState("");
  const [estimate, setEstimate] = useState("");
  async function save() {
    try {
      await request(`/api/models/${model.id}/scripts`, {
        method: "POST",
        body: JSON.stringify({ name: name || null, raw_script: raw, estimated_vram_mib: estimate ? Number(estimate) : null })
      });
      setName("");
      reload();
      toast("Script saved");
    } catch (error) {
      toast(error.message);
    }
  }
  return (
    <details className="scriptEditor">
      <summary><Plus size={14} /> Add loading script</summary>
      <input value={name} onChange={(event) => setName(event.target.value)} placeholder="Optional script name, otherwise autosuggested" />
      <textarea value={raw} onChange={(event) => setRaw(event.target.value)} />
      <div className="toolbar">
        <input value={estimate} onChange={(event) => setEstimate(event.target.value)} placeholder="Optional VRAM estimate MiB" />
        <button className="primary" onClick={save}><Save size={14} /> Save script</button>
      </div>
    </details>
  );
}

function SavedScript({ model, script, start, removeScript, toggleFavorite, reload, toast, onEditingChange }) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(script.name || "");
  const [raw, setRaw] = useState(script.raw_script || "");
  const [estimate, setEstimate] = useState(script.estimated_vram_mib ? String(script.estimated_vram_mib) : "");
  const [saving, setSaving] = useState(false);
  const fitManaged = script.parsed_json?.fit && (!script.parsed_json?.gpu_layers || String(script.parsed_json.gpu_layers).toLowerCase() === "auto");
  const memoryLabel = fitManaged
    ? "VRAM auto-fit"
    : script.parsed_json?.n_cpu_moe
    ? `MoE CPU/RAM (${script.parsed_json.n_cpu_moe} CPU experts)`
    : `${formatMib(script.estimated_vram_mib)} estimated`;

  function reset() {
    setName(script.name || "");
    setRaw(script.raw_script || "");
    setEstimate(script.estimated_vram_mib ? String(script.estimated_vram_mib) : "");
    setEditing(false);
    onEditingChange(false);
  }

  async function save() {
    try {
      setSaving(true);
      await request(`/api/models/${model.id}/scripts/${script.id}`, {
        method: "PATCH",
        body: JSON.stringify({
          name: name || null,
          raw_script: raw,
          estimated_vram_mib: estimate ? Number(estimate) : null
        })
      });
      setEditing(false);
      onEditingChange(false);
      reload();
      toast("Script updated");
    } catch (error) {
      toast(error.message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className={`scriptItem ${editing ? "editing" : ""}`}>
      <div className="scriptRow">
        <IconButton
          icon={Star}
          label={script.is_favorite ? "Remove script from favourites" : "Add script to favourites"}
          className={`favoriteButton ${script.is_favorite ? "active" : ""}`}
          aria-pressed={Boolean(script.is_favorite)}
          onClick={() => toggleFavorite(model, script)}
          disabled={editing}
        />
        <div>
          <strong>{script.name}</strong>
          <span>{script.parsed_json?.ctx_size || "auto"} ctx / {script.parsed_json?.quantization || model.quantization || "quant?"} / {memoryLabel}</span>
        </div>
        <button className="primary" onClick={() => start(script.id)} disabled={editing}><Play size={14} /> Start</button>
        <button className="subtle" onClick={() => { setEditing(true); onEditingChange(true); }} disabled={editing}><Pencil size={14} /> Edit</button>
        <IconButton icon={Trash2} label="Delete script" onClick={() => removeScript(model, script)} disabled={editing} />
      </div>
      {editing && (
        <div className="scriptEditPanel">
          <input value={name} onChange={(event) => setName(event.target.value)} placeholder="Optional script name, otherwise autosuggested" />
          <textarea value={raw} onChange={(event) => setRaw(event.target.value)} />
          <div className="toolbar">
            <input value={estimate} onChange={(event) => setEstimate(event.target.value)} placeholder="Optional VRAM estimate MiB" />
            <button className="primary" onClick={save} disabled={!raw || saving}><Save size={14} /> {saving ? "Saving" : "Save"}</button>
            <button className="subtle" onClick={reset}><X size={14} /> Cancel</button>
          </div>
        </div>
      )}
    </div>
  );
}

function SortableModelBlock({ model, draggingDisabled, children }) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: model.id,
    disabled: draggingDisabled
  });
  const style = {
    transform: CSS.Transform.toString(transform),
    transition
  };
  return (
    <div ref={setNodeRef} style={style} className={`modelBlock ${isDragging ? "dragging" : ""} ${model.source === "ninfer" ? "ninfer" : ""}`}>
      {children({ attributes, listeners, draggingDisabled })}
    </div>
  );
}

function ModelDragPreview({ model }) {
  if (!model) return null;
  return (
    <div className="modelDragPreview">
      <strong>{model.name}</strong>
      <span>{model.quantization || "unknown quant"} / {formatBytes(model.size_bytes)}{model.source === "ninfer" ? " / NInfer" : ""}</span>
    </div>
  );
}

const usageCategories = [
  ["input_tokens", "Input"],
  ["cache_read_tokens", "Cache read"],
  ["cache_write_tokens", "Cache write"],
  ["output_tokens", "Output"],
  ["reasoning_tokens", "Reasoning"]
];

function UsageBreakdown({ stats }) {
  if (!stats) return null;
  return (
    <div className="usageBreakdown">
      {usageCategories.map(([key, label]) => (
        <span className="usageChip" key={key}>
          <small>{label}</small>
          <strong>{formatTokens(stats[key])}</strong>
        </span>
      ))}
    </div>
  );
}

function UsageStatus({ stats }) {
  if (!stats) return null;
  const labels = {
    unpriced: "Unpriced",
    partially_priced: "Partially priced",
    priced: "Priced",
    no_usage: "No usage"
  };
  return (
    <div className="usageFlags">
      {stats.cost_status && <Pill tone={stats.cost_status === "priced" ? "loaded" : stats.cost_status === "no_usage" ? "neutral" : "orphaned"}>{labels[stats.cost_status] || stats.cost_status}</Pill>}
      {stats.reasoning_status === "not_reported" && (
        <Pill
          title="The model does not give a separate reasoning token count. Reasoning is already included in Output and in the estimated cost."
        >
          Reasoning counted as output
        </Pill>
      )}
      {!!stats.missing_rates?.length && <span className="usageMissing">Missing: {stats.missing_rates.join(", ")}</span>}
    </div>
  );
}

function ModelUsage({ model, usage, reload, reloadUsage, toast }) {
  const modelUsage = usage?.models?.find((item) => item.model_id === model.id);
  const stats = modelUsage?.usage;
  const bindingSource = modelUsage?.bindings || model.bindings || [];
  const bindingKey = bindingSource.map((binding) => `${binding.provider_id}\u0000${binding.external_model_id}`).join("\u0001");
  const [draft, setDraft] = useState({});
  const [bindings, setBindings] = useState([]);
  const [providerId, setProviderId] = useState("");
  const [externalModelId, setExternalModelId] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setDraft({
      currency: model.price_currency || "USD",
      input_per_million: model.price_input_per_million || "",
      cache_read_per_million: model.price_cache_read_per_million || "",
      cache_write_per_million: model.price_cache_write_per_million || "",
      output_per_million: model.price_output_per_million || "",
      reasoning_per_million: model.price_reasoning_per_million || ""
    });
  }, [
    model.id,
    model.price_currency,
    model.price_input_per_million,
    model.price_cache_read_per_million,
    model.price_cache_write_per_million,
    model.price_output_per_million,
    model.price_reasoning_per_million
  ]);

  useEffect(() => {
    setBindings(bindingSource.map((binding) => ({
      provider_id: binding.provider_id,
      external_model_id: binding.external_model_id
    })));
  }, [model.id, bindingKey]);

  function addBinding() {
    const provider = providerId.trim();
    const external = externalModelId.trim();
    if (!provider || !external) return;
    if (!bindings.some((binding) => binding.provider_id === provider && binding.external_model_id === external)) {
      setBindings([...bindings, { provider_id: provider, external_model_id: external }]);
    }
    setProviderId("");
    setExternalModelId("");
  }

  async function save() {
    setSaving(true);
    try {
      await request(`/api/models/${model.id}/usage-settings`, { method: "PATCH", body: JSON.stringify({ ...draft, bindings }) });
      await Promise.all([reload(), reloadUsage()]);
      toast("Usage pricing saved");
    } catch (error) {
      toast(error.message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <details className="modelUsage">
      <summary>
        <ChevronRight className="modelUsageChevron" size={14} />
        Usage & pricing
        {stats?.total_tokens ? <span className="usageSummaryHint">{formatTokens(stats.total_tokens)} tokens</span> : null}
      </summary>
      <div className="modelUsageBody">
        <div className="usageInlineSummary">
          <span><small>Usage in selected range</small><strong>{formatTokens(stats?.total_tokens || 0)} tokens</strong></span>
          <span><small>Estimated cost</small><strong>{formatCost(stats?.cost, model.price_currency || "USD")}</strong></span>
          <span><small>Requests</small><strong>{(stats?.requests || 0).toLocaleString()}</strong></span>
        </div>
        <UsageBreakdown stats={stats} />
        <UsageStatus stats={stats} />
        <div className="usagePricingGrid">
          <label>Currency<input value={draft.currency || "USD"} maxLength={8} onChange={(event) => setDraft({ ...draft, currency: event.target.value.toUpperCase() })} /></label>
          <label>Input / MTok<input inputMode="decimal" value={draft.input_per_million || ""} onChange={(event) => setDraft({ ...draft, input_per_million: event.target.value })} placeholder="e.g. 1.00" /></label>
          <label>Cache read / MTok<input inputMode="decimal" value={draft.cache_read_per_million || ""} onChange={(event) => setDraft({ ...draft, cache_read_per_million: event.target.value })} placeholder="e.g. 0.10" /></label>
          <label>Cache write / MTok<input inputMode="decimal" value={draft.cache_write_per_million || ""} onChange={(event) => setDraft({ ...draft, cache_write_per_million: event.target.value })} placeholder="optional" /></label>
          <label>Output / MTok<input inputMode="decimal" value={draft.output_per_million || ""} onChange={(event) => setDraft({ ...draft, output_per_million: event.target.value })} placeholder="e.g. 2.00" /></label>
          <label>Reasoning / MTok<input inputMode="decimal" value={draft.reasoning_per_million || ""} onChange={(event) => setDraft({ ...draft, reasoning_per_million: event.target.value })} placeholder="blank = output" /></label>
        </div>
        <div className="usageBindings">
          <div className="usageSubheading">OpenCode identity bindings</div>
          {!bindings.length && <div className="empty inlineEmpty">Automatic aliases are used first. Add a binding only for an identity shown as Unmapped.</div>}
          {!!bindings.length && <div className="bindingList">{bindings.map((binding) => (
            <span className="bindingPill" key={`${binding.provider_id}/${binding.external_model_id}`}>
              {binding.provider_id} / {binding.external_model_id}
              <button type="button" title="Remove binding" aria-label="Remove binding" onClick={() => setBindings(bindings.filter((item) => item !== binding))}><X size={12} /></button>
            </span>
          ))}</div>}
          <div className="bindingEditor">
            <input value={providerId} onChange={(event) => setProviderId(event.target.value)} placeholder="Provider ID" />
            <input value={externalModelId} onChange={(event) => setExternalModelId(event.target.value)} placeholder="OpenCode model ID" />
            <button type="button" onClick={addBinding} disabled={!providerId.trim() || !externalModelId.trim()}><Plus size={14} /> Add</button>
          </div>
        </div>
        <div className="usageEditorFoot">
          <span>Changing rates recalculates all displayed history.</span>
          <button className="primary" onClick={save} disabled={saving}><Save size={14} /> {saving ? "Saving" : "Save"}</button>
        </div>
      </div>
    </details>
  );
}

function Library({ models, usage, reload, reloadUsage, toast }) {
  const [orderedModels, setOrderedModels] = useState(models);
  const [activeModelId, setActiveModelId] = useState(null);
  const [editingScripts, setEditingScripts] = useState({});
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  );
  const activeModel = orderedModels.find((model) => model.id === activeModelId);
  const favoriteScripts = useMemo(
    () => orderedModels.flatMap((model) => (model.scripts || [])
      .filter((script) => script.is_favorite)
      .map((script) => ({ model, script }))),
    [orderedModels]
  );

  useEffect(() => {
    setOrderedModels(models);
  }, [models]);

  function markScriptEditing(modelId, scriptId, editing) {
    setEditingScripts((current) => {
      const next = { ...current };
      const scriptIds = new Set(next[modelId] || []);
      if (editing) {
        scriptIds.add(scriptId);
      } else {
        scriptIds.delete(scriptId);
      }
      if (scriptIds.size) {
        next[modelId] = Array.from(scriptIds);
      } else {
        delete next[modelId];
      }
      return next;
    });
  }

  async function reorderModels(activeId, overId) {
    if (!overId || activeId === overId) return;
    const oldIndex = orderedModels.findIndex((model) => model.id === activeId);
    const newIndex = orderedModels.findIndex((model) => model.id === overId);
    if (oldIndex < 0 || newIndex < 0) return;
    const previous = orderedModels;
    const next = arrayMove(orderedModels, oldIndex, newIndex);
    setOrderedModels(next);
    try {
      await request("/api/models/order", { method: "PATCH", body: JSON.stringify({ model_ids: next.map((model) => model.id) }) });
      reload();
      toast("Model order saved");
    } catch (error) {
      setOrderedModels(previous);
      toast(error.message);
    }
  }

  async function remove(model) {
    if (!confirm(`Physically delete ${model.name}?`)) return;
    try {
      await request(`/api/models/${model.id}`, { method: "DELETE" });
      reload();
      toast("Model deleted");
    } catch (error) {
      toast(error.message);
    }
  }
  async function start(scriptId) {
    try {
      await request("/api/runs/start", { method: "POST", body: JSON.stringify({ script_id: scriptId }) });
      reload();
      toast("Loading started");
    } catch (error) {
      toast(error.message);
    }
  }
  async function removeScript(model, script) {
    if (!confirm(`Delete loading script ${script.name}?`)) return;
    try {
      await request(`/api/models/${model.id}/scripts/${script.id}`, { method: "DELETE" });
      reload();
      toast("Script deleted");
    } catch (error) {
      toast(error.message);
    }
  }
  async function toggleFavorite(model, script) {
    try {
      await request(`/api/models/${model.id}/scripts/${script.id}/favorite`, {
        method: "PATCH",
        body: JSON.stringify({ is_favorite: !script.is_favorite })
      });
      reload();
      toast(script.is_favorite ? "Removed from favourites" : "Added to favourites");
    } catch (error) {
      toast(error.message);
    }
  }
  return (
    <>
      <section className="band quickStartBand">
        <div className="sectionTitle">
          <Star size={16} />
          <h2>Quick start favourite scripts</h2>
          {!!favoriteScripts.length && <span className="sectionCount">{favoriteScripts.length}</span>}
        </div>
        {!favoriteScripts.length && (
          <div className="empty quickStartEmpty">Star a loading script in the model library to keep it within easy reach.</div>
        )}
        {!!favoriteScripts.length && (
          <div className="quickStartGrid">
            {favoriteScripts.map(({ model, script }) => (
              <div className="quickStartItem" key={script.id}>
                <Star size={15} fill="currentColor" aria-hidden="true" />
                <div>
                  <strong>{script.name}</strong>
                  <span className="quickStartModel">
                    {model.name}
                    {model.source === "ninfer" && <Pill tone="ninfer">NInfer</Pill>}
                  </span>
                </div>
                <button className="primary" onClick={() => start(script.id)}>
                  <Play size={14} /> Start
                </button>
              </div>
            ))}
          </div>
        )}
      </section>
      <section className="band modelLibraryBand">
        <div className="sectionTitle">
          <HardDrive size={16} />
          <h2>Model Library</h2>
        </div>
        {!orderedModels.length && <div className="empty">Downloaded and imported models will appear here.</div>}
      <DndContext
        sensors={sensors}
        collisionDetection={closestCenter}
        onDragStart={(event) => setActiveModelId(event.active.id)}
        onDragCancel={() => setActiveModelId(null)}
        onDragEnd={(event) => {
          setActiveModelId(null);
          reorderModels(event.active.id, event.over?.id);
        }}
      >
        <SortableContext items={orderedModels.map((model) => model.id)} strategy={verticalListSortingStrategy}>
          {orderedModels.map((model) => {
            const draggingDisabled = Boolean(editingScripts[model.id]?.length);
            return (
              <SortableModelBlock key={model.id} model={model} draggingDisabled={draggingDisabled}>
                {({ attributes, listeners }) => (
                  <>
                    <div className="modelHead">
                      <button
                        className="dragHandle"
                        title={draggingDisabled ? "Finish editing this model's script before sorting" : "Drag to reorder model"}
                        aria-label={`Drag to reorder ${model.name}`}
                        disabled={draggingDisabled}
                        {...attributes}
                        {...listeners}
                      >
                        <GripVertical size={15} />
                      </button>
                      <div>
                        <span className="modelTitleLine">
                          <strong>{model.name}</strong>
                          {model.source === "ninfer" && <Pill tone="ninfer">NInfer</Pill>}
                        </span>
                        <span>
                          {model.quantization || "unknown quant"} / {formatBytes(model.size_bytes)}
                          {model.shard_count > 1 ? ` / ${model.shard_count} parts` : ""} / {model.path}
                        </span>
                      </div>
                      <IconButton icon={Trash2} label="Delete model" onClick={() => remove(model)} />
                    </div>
                    <ModelUsage model={model} usage={usage} reload={reload} reloadUsage={reloadUsage} toast={toast} />
                    <details className="modelScripts">
                      <summary>
                        <ChevronRight className="modelScriptsChevron" size={14} />
                        Loading scripts ({model.scripts?.length || 0})
                      </summary>
                      <div className="scripts">
                        {model.scripts?.map((script) => (
                          <SavedScript
                            key={script.id}
                            model={model}
                            script={script}
                            start={start}
                            removeScript={removeScript}
                            toggleFavorite={toggleFavorite}
                            reload={reload}
                            toast={toast}
                            onEditingChange={(editing) => markScriptEditing(model.id, script.id, editing)}
                          />
                        ))}
                      </div>
                      <ScriptEditor model={model} reload={reload} toast={toast} />
                    </details>
                  </>
                )}
              </SortableModelBlock>
            );
          })}
        </SortableContext>
        <DragOverlay>
          <ModelDragPreview model={activeModel} />
        </DragOverlay>
      </DndContext>
      </section>
    </>
  );
}

function UsagePanel({ usage, range, setRange, page, setPage, reloadUsage, reload, toast }) {
  if (!usage) {
    return <section className="band usageBand"><div className="sectionTitle"><BarChart3 size={16} /><h2>Usage & Cost</h2></div><div className="empty">Loading OpenCode usage…</div></section>;
  }
  if (!usage.available) {
    return (
      <section className="band usageBand">
        <div className="sectionTitle"><BarChart3 size={16} /><h2>Usage & Cost</h2><div className="spacer" /><button className="subtle" onClick={reloadUsage}><RefreshCw size={14} /> Refresh</button></div>
        <div className="empty inlineEmpty"><strong>OpenCode usage unavailable.</strong> {usage.error || "Check the database path in Runtime Settings."}</div>
      </section>
    );
  }
  const summary = usage.summary || {};
  const recent = usage.recent_task;
  const history = usage.history || { items: [], total: 0, page: 0, page_size: 8 };
  const pages = Math.max(1, Math.ceil((history.total || 0) / (history.page_size || 8)));
  const currencies = Array.from(new Set((usage.models || []).map((item) => item.price_currency || "USD")));
  const currency = currencies.length === 1 ? currencies[0] : "USD";
  const mixedCurrencies = currencies.length > 1;
  const displayCost = (value) => mixedCurrencies ? `${value || "0"} (mixed currencies)` : formatCost(value, currency);
  async function assignIdentity(identity, targetModelId) {
    if (!targetModelId) return;
    const target = usage.models?.find((item) => item.model_id === targetModelId);
    if (!target) return;
    const bindings = [...(target.bindings || []), { provider_id: identity.provider_id, external_model_id: identity.external_model_id }];
    try {
      await request(`/api/models/${targetModelId}/usage-settings`, {
        method: "PATCH",
        body: JSON.stringify({
          currency: target.price_currency,
          input_per_million: target.price_input_per_million,
          cache_read_per_million: target.price_cache_read_per_million,
          cache_write_per_million: target.price_cache_write_per_million,
          output_per_million: target.price_output_per_million,
          reasoning_per_million: target.price_reasoning_per_million,
          bindings
        })
      });
      await Promise.all([reload(), reloadUsage()]);
      toast("OpenCode identity assigned");
    } catch (error) {
      toast(error.message);
    }
  }
  return (
    <section className="band usageBand">
      <div className="sectionTitle">
        <BarChart3 size={16} />
        <h2>Usage & Cost</h2>
        <div className="spacer" />
        <select className="usageRange" value={range} onChange={(event) => { setRange(event.target.value); setPage(0); }} aria-label="Usage date range">
          <option value="7d">7 days</option><option value="30d">30 days</option><option value="all">All</option>
        </select>
        <button className="subtle" onClick={reloadUsage} title="Refresh usage"><RefreshCw size={14} /></button>
      </div>
      <div className="usageSummary">
        <span><small>Processed tokens</small><strong>{formatTokens(summary.total_tokens)}</strong></span>
        <span><small>Estimated cost</small><strong>{displayCost(summary.cost)}</strong></span>
        <span><small>Requests</small><strong>{(summary.requests || 0).toLocaleString()}</strong></span>
        <span><small>Main / subagents</small><strong>{formatTokens(recent?.main?.total_tokens || 0)} / {formatTokens(recent?.subagents?.total_tokens || 0)}</strong></span>
      </div>
      <UsageBreakdown stats={summary} />
      <UsageStatus stats={summary} />
      {recent && (
        <details className="usageTask" open>
          <summary><ChevronRight className="usageTaskChevron" size={14} /><span><strong>{recent.title}</strong><small>{recent.model_names?.join(", ") || "Unknown model"} · {formatDate(recent.updated_at)}</small></span><b>{formatTokens(recent.usage?.total_tokens || 0)}</b></summary>
          <div className="usageTaskBody">
            <div className="usageTaskTotals"><span>Main <strong>{formatTokens(recent.main?.total_tokens || 0)}</strong></span><span>Subagents <strong>{formatTokens(recent.subagents?.total_tokens || 0)}</strong></span><span>Cost <strong>{displayCost(recent.usage?.cost)}</strong></span></div>
            <UsageStatus stats={recent.usage} />
            <div className="usageChildren">
              <div className="usageChild"><span><strong>Main session</strong><small>{recent.main?.model_names?.join(", ") || "No recorded responses"}</small></span><b>{formatTokens(recent.main?.total_tokens || 0)}</b></div>
              {(recent.children || []).map((child) => <div className="usageChild" key={child.session_id}><span><strong>{child.title}</strong><small>{child.agent || child.usage?.model_names?.join(", ") || "Subagent"}</small></span><b>{formatTokens(child.usage?.total_tokens || 0)}</b></div>)}
            </div>
          </div>
        </details>
      )}
      {!recent && <div className="empty inlineEmpty">No usage yet.</div>}
      <details className="usageHistory">
        <summary><ChevronRight className="usageHistoryChevron" size={14} /> History ({history.total || 0} tasks)</summary>
        <div className="table usageTable">
          <div className="thead"><span>Updated</span><span>Task / model</span><span>Main</span><span>Subagents</span><span>Total</span><span>Cost</span></div>
          {!history.items?.length && <div className="empty">No usage in this range.</div>}
          {(history.items || []).map((item) => (
            <div className="row" key={item.session_id}>
              <span>{formatDate(item.updated_at)}</span>
              <span className="usageHistoryTask"><strong title={item.title}>{item.title}</strong><small title={item.model_names?.join(", ")}>{item.model_names?.join(", ") || "Unknown model"}</small></span>
              <span>{formatTokens(item.main?.total_tokens || 0)}</span>
              <span>{formatTokens(item.subagents?.total_tokens || 0)}</span>
              <span>{formatTokens(item.usage?.total_tokens || 0)}</span>
              <span>{displayCost(item.usage?.cost)}</span>
            </div>
          ))}
        </div>
        {history.total > (history.page_size || 8) && <div className="pager"><button onClick={() => setPage(Math.max(0, page - 1))} disabled={page === 0}>Previous</button><span>Page {page + 1} / {pages}</span><button onClick={() => setPage(Math.min(pages - 1, page + 1))} disabled={page >= pages - 1}>Next</button></div>}
      </details>
      {!!usage.unmapped?.length && <details className="usageUnmapped"><summary><ChevronRight className="usageHistoryChevron" size={14} /> Unmapped identities ({usage.unmapped.length})</summary><div className="unmappedList">{usage.unmapped.map((item) => <div className="unmappedRow" key={`${item.provider_id}/${item.external_model_id}`}><span className="bindingPill unmapped">{item.provider_id} / {item.external_model_id} · {formatTokens(item.total_tokens)}</span><select defaultValue="" aria-label={`Assign ${item.external_model_id} to a model`} onChange={(event) => assignIdentity(item, event.target.value)}><option value="">Assign to model…</option>{(usage.models || []).map((model) => <option key={model.model_id} value={model.model_id}>{model.model_name}</option>)}</select></div>)}</div><small className="usageHelp">Assignment is saved as an explicit provider/model binding.</small></details>}
    </section>
  );
}

function Runs({ runs, models, reload, toast }) {
  const deletableStatuses = new Set(["aborted", "failed", "unloaded", "exited"]);
  const inactiveCount = runs.filter((run) => deletableStatuses.has(run.status)).length;
  const protectedCount = runs.length - inactiveCount;
  const modelNames = useMemo(() => new Map(models.map((model) => [model.id, model.name])), [models]);
  function statusMessage(run) {
    if (run.status_message) return run.status_message;
    if (run.status === "failed" && run.error) return run.error;
    if (run.status === "aborted") return "Aborted: server process was stopped.";
    if (run.status === "unloaded") return "Unloaded: server process was stopped.";
    if (run.status === "loaded") return "Loaded: server is running.";
    if (run.status === "loading") return "Loading: waiting for startup output and health check.";
    return "Waiting for server status...";
  }
  async function action(run, kind) {
    try {
      await request(`/api/runs/${run.id}/${kind}`, { method: "POST" });
      reload();
      toast(kind === "abort" ? "Run aborted" : "Run unloaded");
    } catch (error) {
      toast(error.message);
    }
  }
  async function remove(run) {
    if (!confirm("Delete this terminal session from history?")) return;
    try {
      await request(`/api/runs/${run.id}`, { method: "DELETE" });
      reload();
      toast("Run history deleted");
    } catch (error) {
      toast(error.message);
    }
  }
  async function clearHistory() {
    const suffix = protectedCount ? ` ${protectedCount} active/loading session${protectedCount === 1 ? "" : "s"} will stay.` : "";
    if (!confirm(`Delete ${inactiveCount} historical terminal session${inactiveCount === 1 ? "" : "s"}?${suffix}`)) return;
    try {
      const result = await request("/api/runs/history", { method: "DELETE" });
      reload();
      toast(`Deleted ${result.deleted} historical session${result.deleted === 1 ? "" : "s"}`);
    } catch (error) {
      toast(error.message);
    }
  }
  return (
    <section className="band">
      <div className="sectionTitle">
        <Activity size={16} />
        <h2>Active Runs & Terminal</h2>
        <div className="spacer" />
        <button className="subtle" onClick={clearHistory} disabled={!inactiveCount}>
          <Trash2 size={14} /> Clear history
        </button>
      </div>
      {!runs.length && <div className="empty">No process history yet.</div>}
      {runs.map((run) => {
        const elapsed = run.ended_at ? run.ended_at - run.started_at : Date.now() / 1000 - run.started_at;
        const message = statusMessage(run);
        const terminalText = compactLogTail(run.log_tail) || `[loader] ${message}`;
        const modelName = run.model_name || modelNames.get(run.model_id) || "Unknown model";
        const scriptName = run.script_name || "Deleted script";
        return (
          <div className="runBlock" key={run.id}>
            <div className="runHead">
              <Pill tone={run.status}>{run.status}</Pill>
              <span className="runModel" title={modelName}>
                <Cpu size={14} />
                <span>
                  <strong>{modelName}</strong>
                  <small>Start script: {scriptName}</small>
                </span>
              </span>
              {["loading", "orphaned"].includes(run.status) && <button onClick={() => action(run, "abort")}><Square size={14} /> Abort</button>}
              {run.status === "loaded" && <button onClick={() => action(run, "unload")}><Square size={14} /> Unload</button>}
              {deletableStatuses.has(run.status) && <IconButton icon={Trash2} label="Delete terminal history" onClick={() => remove(run)} />}
            </div>
            <details className="runDetails">
              <summary>
                <ChevronRight className="runDetailsChevron" size={14} />
                Run details
              </summary>
              <div className="runDetailsGrid">
                <span><strong>PID</strong>{run.pid || "n/a"}</span>
                <span><strong>Address</strong>{run.host ? `${run.host}${run.port ? `:${run.port}` : ""}` : "n/a"}</span>
                <span><strong>Elapsed</strong>{formatDuration(elapsed)}</span>
              </div>
            </details>
            <div className={`runStatusLine ${run.status}`}>
              <span className="statusDot" />
              <span>{message}</span>
            </div>
            <pre className="terminal">{terminalText}</pre>
          </div>
        );
      })}
    </section>
  );
}

function Benchmarks({ presets, models, runs, reload, toast }) {
  const scripts = useMemo(() => models.flatMap((model) => (model.scripts || []).map((script) => ({ ...script, modelName: model.name }))), [models]);
  const loadedScriptIds = useMemo(() => new Set(runs.filter((run) => run.status === "loaded").map((run) => run.script_id)), [runs]);
  const loadedScripts = useMemo(() => scripts.filter((script) => loadedScriptIds.has(script.id)), [scripts, loadedScriptIds]);
  const [scriptId, setScriptId] = useState("");
  const [presetId, setPresetId] = useState("small");
  const [prompt, setPrompt] = useState("");
  const [outputTokens, setOutputTokens] = useState("");
  const [history, setHistory] = useState([]);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [historyPage, setHistoryPage] = useState(0);
  const [modelFilter, setModelFilter] = useState("");
  const [scriptFilter, setScriptFilter] = useState("");
  const [presetFilter, setPresetFilter] = useState("");
  const pageSize = 7;
  const preset = presets.find((item) => item.id === presetId);
  const filteredScripts = useMemo(() => scripts.filter((script) => !modelFilter || script.model_id === modelFilter), [scripts, modelFilter]);
  const pages = Math.max(1, Math.ceil(historyTotal / pageSize));
  const selectedScript = loadedScripts.find((script) => script.id === scriptId);

  const loadHistory = useCallback(async () => {
    const params = new URLSearchParams({
      active_only: String(!historyOpen),
      limit: String(pageSize),
      offset: String(historyOpen ? historyPage * pageSize : 0)
    });
    if (modelFilter) params.set("model_id", modelFilter);
    if (scriptFilter) params.set("script_id", scriptFilter);
    if (presetFilter) params.set("preset_id", presetFilter);
    const data = await request(`/api/benchmarks/history?${params.toString()}`);
    setHistory(data.items || []);
    setHistoryTotal(data.total || 0);
  }, [historyOpen, historyPage, modelFilter, scriptFilter, presetFilter]);

  useEffect(() => {
    loadHistory().catch((error) => toast(error.message));
    const timer = window.setInterval(() => loadHistory().catch(() => {}), 2500);
    return () => window.clearInterval(timer);
  }, [loadHistory, toast]);

  useEffect(() => {
    setHistoryPage(0);
  }, [historyOpen, modelFilter, scriptFilter, presetFilter]);

  useEffect(() => {
    if (!preset) return;
    setPrompt(preset.prompt || "");
    setOutputTokens(String(preset.output_tokens || ""));
  }, [preset]);

  useEffect(() => {
    if (loadedScripts.length === 1) {
      setScriptId(loadedScripts[0].id);
      return;
    }
    if (scriptId && !loadedScripts.some((script) => script.id === scriptId)) {
      setScriptId("");
    }
  }, [loadedScripts, scriptId]);

  async function run() {
    try {
      await request("/api/benchmarks", {
        method: "POST",
        body: JSON.stringify({
          script_id: scriptId,
          preset_id: presetId,
          prompt,
          output_tokens: Number(outputTokens) || undefined
        })
      });
      reload();
      loadHistory();
      toast("Benchmark started");
    } catch (error) {
      toast(error.message);
    }
  }
  return (
    <section className="band">
      <div className="sectionTitle">
        <BarChart3 size={16} />
        <h2>Benchmark Dashboard</h2>
        <div className="spacer" />
        <button className="subtle" onClick={() => setHistoryOpen(!historyOpen)}>
          {historyOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          {historyOpen ? "Show active only" : "Open history"}
        </button>
      </div>
      <div className="toolbar wrap">
        <select value={scriptId} onChange={(event) => setScriptId(event.target.value)}>
          <option value="">{loadedScripts.length ? "Select loaded script version" : "No loaded script available"}</option>
          {loadedScripts.map((script) => <option key={script.id} value={script.id}>{script.modelName} / {script.name}</option>)}
        </select>
        <select value={presetId} onChange={(event) => setPresetId(event.target.value)}>
          {presets.map((item) => <option key={item.id} value={item.id}>{item.name}: {item.prompt_tokens} in / {item.output_tokens} out</option>)}
        </select>
        <input className="tokenInput" type="number" min="1" value={outputTokens} onChange={(event) => setOutputTokens(event.target.value)} title="Expected output tokens" />
        <button className="primary" onClick={run} disabled={!scriptId}><Play size={14} /> Run</button>
      </div>
      {selectedScript && <div className="selectionHint">Ready: <strong>{selectedScript.modelName}</strong><span>{selectedScript.name}</span></div>}
      {!loadedScripts.length && <div className="empty inlineEmpty">Load a model script first, then this dashboard can run benchmarks against it.</div>}
      {loadedScripts.length > 1 && !scriptId && <div className="empty inlineEmpty">Choose one loaded script version to enable Run.</div>}
      {preset && <textarea className="promptPreview" value={prompt} onChange={(event) => setPrompt(event.target.value)} />}
      <div className="toolbar compactFilters">
        <select value={modelFilter} onChange={(event) => setModelFilter(event.target.value)}>
          <option value="">All models</option>
          {models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
        </select>
        <select value={scriptFilter} onChange={(event) => setScriptFilter(event.target.value)}>
          <option value="">All scripts</option>
          {filteredScripts.map((script) => <option key={script.id} value={script.id}>{script.name}</option>)}
        </select>
        <select value={presetFilter} onChange={(event) => setPresetFilter(event.target.value)}>
          <option value="">All presets</option>
          {presets.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
        </select>
      </div>
      <div className="table">
        <div className="thead benchGrid"><span>Model</span><span>Preset</span><span>Status</span><span>FTTT</span><span>Prefill</span><span>Generation</span><span>Average</span></div>
        {!history.length && <div className="empty">{historyOpen ? "No benchmark history matches these filters." : "No active benchmarks. Select a loaded script and press Run to start one."}</div>}
        {history.map((item) => (
          <details className="benchItem" key={item.id}>
            <summary className="row benchGrid">
              <span className="benchModel">
                <strong title={item.script_name || "Script unavailable"}>{item.model_name || "unknown model"}</strong>
                <small title={item.script_name || ""}>{item.script_name || "script unavailable"}</small>
              </span>
              <strong>{item.preset_id || "custom"}</strong>
              <Pill tone={item.status}>{item.status}</Pill>
              <span>{item.fttt_ms ? `${item.fttt_ms.toFixed(0)} ms` : "n/a"}</span>
              <span>{item.prefill_tps ? `${item.prefill_tps.toFixed(2)} tok/s` : "n/a"}</span>
              <span>{item.generation_tps ? `${item.generation_tps.toFixed(2)} tok/s` : "n/a"}</span>
              <span>{item.average_tps ? `${item.average_tps.toFixed(2)} tok/s` : "n/a"}</span>
            </summary>
            <pre className="terminal small">{item.raw_log || "No benchmark log captured yet."}</pre>
          </details>
        ))}
      </div>
      {historyOpen && historyTotal > pageSize && (
        <div className="pager">
          <button onClick={() => setHistoryPage(Math.max(0, historyPage - 1))} disabled={historyPage === 0}>Previous</button>
          <span>Page {historyPage + 1} / {pages}</span>
          <button onClick={() => setHistoryPage(Math.min(pages - 1, historyPage + 1))} disabled={historyPage >= pages - 1}>Next</button>
        </div>
      )}
    </section>
  );
}

function App() {
  const [settings, setSettings] = useState({});
  const [telemetry, setTelemetry] = useState(null);
  const [models, setModels] = useState([]);
  const [downloads, setDownloads] = useState([]);
  const [runs, setRuns] = useState([]);
  const [presets, setPresets] = useState([]);
  const [usage, setUsage] = useState(null);
  const [usageRange, setUsageRange] = useState("all");
  const [usagePage, setUsagePage] = useState(0);
  const [toastText, setToastText] = useState("");

  const toast = useCallback((message) => {
    setToastText(message);
    window.setTimeout(() => setToastText(""), 3000);
  }, []);

  const reload = useCallback(async () => {
    const [settingsData, telemetryData, modelData, downloadData, runData, presetData] = await Promise.all([
      request("/api/settings"),
      request("/api/system/telemetry"),
      request("/api/models"),
      request("/api/downloads"),
      request("/api/runs"),
      request("/api/benchmarks")
    ]);
    setSettings(settingsData);
    setTelemetry(telemetryData);
    setModels(modelData);
    setDownloads(downloadData);
    setRuns(runData);
    setPresets(presetData);
  }, []);

  const reloadUsage = useCallback(async () => {
    const params = new URLSearchParams({ range: usageRange, page: String(usagePage) });
    const usageData = await request(`/api/usage?${params.toString()}`);
    setUsage(usageData);
  }, [usageRange, usagePage]);

  useEffect(() => {
    reload().catch((error) => toast(error.message));
    const timer = window.setInterval(() => reload().catch(() => {}), 5000);
    const ws = new WebSocket(`ws://${window.location.host}/ws/events`);
    ws.onmessage = () => reload().catch(() => {});
    return () => {
      window.clearInterval(timer);
      ws.close();
    };
  }, [reload, toast]);

  useEffect(() => {
    reloadUsage().catch((error) => setUsage({ available: false, error: error.message, history: { items: [], total: 0, page: 0, page_size: 8 }, unmapped: [] }));
    const timer = window.setInterval(() => reloadUsage().catch(() => {}), 10000);
    return () => window.clearInterval(timer);
  }, [reloadUsage]);

  return (
    <>
      <TopTelemetry telemetry={telemetry} usage={usage} refresh={() => reload().catch((error) => toast(error.message))} />
      <main>
        <Settings settings={settings} setSettings={setSettings} toast={toast} />
        <div className="grid two dashboardGrid">
          <div className="stack">
            <Discover toast={toast} reload={reload} telemetry={telemetry} />
            <Library models={models} usage={usage} reload={reload} reloadUsage={reloadUsage} toast={toast} />
          </div>
          <div className="stack">
            <ImportModel reload={reload} toast={toast} settings={settings} />
            <Downloads downloads={downloads} reload={reload} toast={toast} />
            <UsagePanel usage={usage} range={usageRange} setRange={setUsageRange} page={usagePage} setPage={setUsagePage} reload={reload} reloadUsage={() => reloadUsage().catch(() => {})} toast={toast} />
            <Runs runs={runs} models={models} reload={reload} toast={toast} />
            <Benchmarks presets={presets} models={models} runs={runs} reload={reload} toast={toast} />
          </div>
        </div>
      </main>
      {toastText && <div className="toast">{toastText}</div>}
    </>
  );
}

createRoot(document.getElementById("root")).render(<App />);
