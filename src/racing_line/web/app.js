"use strict";

const TYRES = {
  soft: { label: "Soft", short: "S", colour: "#ff4057" },
  medium: { label: "Medium", short: "M", colour: "#ffd166" },
  hard: { label: "Hard", short: "H", colour: "#e9edf5" },
  intermediate: { label: "Intermediate", short: "I", colour: "#58d68d" },
  wet: { label: "Full Wet", short: "W", colour: "#4aa3ff" }
};

const WEATHER = {
  dry: { label: "Dry", rain: 0, temperature: 25, colour: "#ffd166" },
  damp: { label: "Damp", rain: 15, temperature: 24, colour: "#94d9d4" },
  light_rain: { label: "Light rain", rain: 40, temperature: 20, colour: "#55b8e8" },
  wet: { label: "Wet", rain: 70, temperature: 16, colour: "#377ed1" },
  heavy_rain: { label: "Heavy rain", rain: 100, temperature: 14, colour: "#715ee6" }
};

const MAX_LAPS = 25;
const MIN_ZOOM = 1;
const MAX_ZOOM = 12;

const form = document.getElementById("simulationForm");
const circuitSelect = document.getElementById("circuitSelect");
const lapsInput = document.getElementById("laps");
const initialTyre = document.getElementById("initialTyre");
const pitStopRows = document.getElementById("pitStopRows");
const weatherRows = document.getElementById("weatherRows");
const addPitStopButton = document.getElementById("addPitStop");
const addWeatherButton = document.getElementById("addWeatherChange");
const dryPresetButton = document.getElementById("dryPreset");
const rainPresetButton = document.getElementById("rainPreset");
const strategyTimeline = document.getElementById("strategyTimeline");
const strategyStatus = document.getElementById("strategyStatus");
const runButton = document.getElementById("runButton");
const runButtonText = document.getElementById("runButtonText");
const statusMessage = document.getElementById("statusMessage");
const trackPanel = document.getElementById("trackPanel");
const trackStage = document.getElementById("trackStage");
const trackCanvas = document.getElementById("trackCanvas");
const trackContext = trackCanvas.getContext("2d");
const chartCanvas = document.getElementById("lapChart");
const chartContext = chartCanvas.getContext("2d");
const speedTraceCanvas = document.getElementById("speedTraceChart");
const speedTraceContext = speedTraceCanvas?.getContext("2d") || null;
const emptyState = document.getElementById("emptyState");
const mapOverlay = document.getElementById("mapOverlay");
const replayButton = document.getElementById("replayButton");
const replayPosition = document.getElementById("replayPosition");
const liveBadge = document.getElementById("liveBadge");
const liveText = document.getElementById("liveText");
const lapResults = document.getElementById("lapResults");
const mapTooltip = document.getElementById("mapTooltip");
const zoomLevel = document.getElementById("zoomLevel");
const fullscreenButton = document.getElementById("fullscreenButton");
const navigationLinks = [...document.querySelectorAll("[data-view-target]")];
const appViews = [...document.querySelectorAll("[data-view]")];
const currentViewLabel = document.getElementById("currentViewLabel");
const resultsCircuitLabel = document.getElementById("resultsCircuitLabel");
const resultsEyebrow = document.getElementById("resultsEyebrow");
const savedRunContext = document.getElementById("savedRunContext");
const savedRunContextText = document.getElementById("savedRunContextText");
const rerunSavedButton = document.getElementById("rerunSavedButton");
const sidebarCircuitName = document.getElementById("sidebarCircuitName");
const sidebarCircuitLocation = document.getElementById("sidebarCircuitLocation");
const footerCircuit = document.getElementById("footerCircuit");
const circuitMeta = document.getElementById("circuitMeta");
const historyNavCount = document.getElementById("historyNavCount");
const historyRunCount = document.getElementById("historyRunCount");
const historyCircuitCount = document.getElementById("historyCircuitCount");
const historyBestLap = document.getElementById("historyBestLap");
const historyMessage = document.getElementById("historyMessage");
const historyPanel = document.getElementById("historyPanel");
const historyTableBody = document.getElementById("historyTableBody");
const clearHistoryButton = document.getElementById("clearHistoryButton");
const tyreWearCanvas = document.getElementById("tyreWearChart");
const tyreWearContext = tyreWearCanvas?.getContext("2d") || null;
const tyreReviewMessage = document.getElementById("tyreReviewMessage");
const tyreOverview = document.getElementById("tyreOverview");
const tyreChartWrap = document.getElementById("tyreChartWrap");
const tyreReviewDetails = document.getElementById("tyreReviewDetails");
const tyreStintCards = document.getElementById("tyreStintCards");
const tyreCauseList = document.getElementById("tyreCauseList");
const tyreRecommendationList = document.getElementById("tyreRecommendationList");
const comparisonPanel = document.getElementById("comparisonPanel");
const comparisonTitle = document.getElementById("comparisonTitle");
const comparisonMessage = document.getElementById("comparisonMessage");
const comparisonValues = {
  fastest: document.getElementById("comparisonFastestDelta"),
  average: document.getElementById("comparisonAverageDelta"),
  total: document.getElementById("comparisonTotalDelta"),
  speed: document.getElementById("comparisonSpeedDelta"),
  incidents: document.getElementById("comparisonIncidentsDelta"),
  clearance: document.getElementById("comparisonClearanceDelta")
};
const consistencyRating = document.getElementById("consistencyRating");
const consistencyStdDev = document.getElementById("consistencyStdDev");
const consistencyRange = document.getElementById("consistencyRange");
const consistencyTrend = document.getElementById("consistencyTrend");
const layerInputs = {
  track: document.getElementById("showTrack"),
  optimized: document.getElementById("showOptimized"),
  driven: document.getElementById("showDriven"),
  heatmap: document.getElementById("showHeatmap")
};
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

let simulationData = null;
let lastPlan = null;
let replayFrame = 0;
let replayStartedAt = 0;
let replayProgress = 0;
let replayRunning = false;
let chartLapTimes = [];
let tyreLapPoints = [];
let rowSequence = 0;
let tooltipPinned = false;
let hoveredTelemetryIndex = -1;
let lastGeometry = null;
let maximumLaps = MAX_LAPS;
let defaultLaps = 10;
let circuitCatalog = [];
let historyEntries = [];
let historyLoading = false;
let historyErrorMessage = "";
let historyDetailLoadingId = "";
let viewingSavedEntry = null;

const mapView = { zoom: 1, panX: 0, panY: 0 };
const activePointers = new Map();
let dragOrigin = null;
let dragMoved = false;
let pinchState = null;

function finiteNumber(value, fallback = 0) {
  if (value === null || value === undefined || value === "") return fallback;
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function firstFinite(...values) {
  for (const value of values) {
    const number = Number(value);
    if (Number.isFinite(number)) return number;
  }
  return NaN;
}

function numericArray(value) {
  if (!Array.isArray(value)) return [];
  return value.map(item => Number(item)).filter(Number.isFinite);
}

function firstDefined(...values) {
  return values.find(value => value !== undefined && value !== null);
}

function normalizeCircuit(raw) {
  if (!raw || typeof raw !== "object") return null;
  const id = String(firstDefined(raw.id, raw.circuit_id, "")).trim();
  const name = String(firstDefined(raw.name, raw.track_name, id)).trim();
  if (!id || !name) return null;
  return {
    id,
    name,
    location: String(firstDefined(raw.location, "")).trim(),
    countryCode: String(firstDefined(raw.country_code, raw.countryCode, "")).trim().toUpperCase(),
    lengthM: finiteNumber(firstDefined(raw.length_m, raw.lengthM), NaN)
  };
}

function circuitDetails(circuit, includeLength = true) {
  if (!circuit) return "F1 circuit";
  const details = [];
  if (circuit.location) details.push(circuit.location);
  if (circuit.countryCode) details.push(circuit.countryCode);
  if (includeLength && Number.isFinite(circuit.lengthM) && circuit.lengthM > 0) {
    details.push(`${(circuit.lengthM / 1000).toFixed(3)} km`);
  }
  return details.length ? details.join(" \u00b7 ") : "Formula 1 circuit";
}

function selectedCircuit() {
  return circuitCatalog.find(circuit => circuit.id === circuitSelect.value)
    || normalizeCircuit({ id: circuitSelect.value || "gb-1948", name: circuitSelect.selectedOptions[0]?.textContent || "Silverstone Circuit", location: "Silverstone", country_code: "GB", length_m: 5891 });
}

function updateCircuitContext() {
  const circuit = selectedCircuit();
  if (!circuit) return;
  const details = circuitDetails(circuit);
  circuitMeta.textContent = details;
  sidebarCircuitName.textContent = circuit.name;
  sidebarCircuitLocation.textContent = details;
  footerCircuit.textContent = `${circuit.name} \u00b7 ${circuitDetails(circuit, false)}`;
  if (!simulationData) {
    document.getElementById("trackTitle").textContent = circuit.name;
    trackCanvas.setAttribute("aria-label", `${circuit.name} circuit map. Run a simulation to display the optimized and driven racing lines.`);
  }
}

function populateCircuitSelect(circuits, defaultCircuitId) {
  const fallback = normalizeCircuit({
    id: "gb-1948",
    name: "Silverstone Circuit",
    location: "Silverstone",
    country_code: "GB"
  });
  circuitCatalog = circuits.map(normalizeCircuit).filter(Boolean);
  if (!circuitCatalog.length && fallback) circuitCatalog = [fallback];

  const previous = circuitSelect.value;
  circuitSelect.replaceChildren();
  const displayCircuits = [...circuitCatalog].sort((a, b) => a.name.localeCompare(b.name, undefined, {
    sensitivity: "base",
    numeric: true
  }));
  for (const circuit of displayCircuits) {
    const option = document.createElement("option");
    option.value = circuit.id;
    const suffix = [circuit.location, circuit.countryCode].filter(Boolean).join(", ");
    option.textContent = suffix ? `${circuit.name} \u2014 ${suffix}` : circuit.name;
    circuitSelect.append(option);
  }
  const requested = [previous, defaultCircuitId, "gb-1948"].find(id => circuitCatalog.some(circuit => circuit.id === id));
  circuitSelect.value = requested || displayCircuits[0].id;
  updateCircuitContext();
}

async function loadConfiguration() {
  const response = await fetch("/api/config", { headers: { "Accept": "application/json" } });
  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error("The circuit catalogue returned invalid JSON.");
  }
  if (!response.ok || !payload || typeof payload !== "object") {
    throw new Error(typeof payload?.error === "string" ? payload.error : "The circuit catalogue could not be loaded.");
  }

  maximumLaps = Math.max(1, Math.round(finiteNumber(payload.maximum_laps, MAX_LAPS)));
  defaultLaps = Math.max(1, Math.min(maximumLaps, Math.round(finiteNumber(payload.default_laps, 10))));
  lapsInput.max = String(maximumLaps);
  lapsInput.value = String(defaultLaps);
  document.getElementById("runHint").textContent = `Choose 1\u2013${maximumLaps} laps. Tyre and weather changes are applied from their selected lap.`;
  document.getElementById("requestedLaps").textContent = `of ${defaultLaps} ${defaultLaps === 1 ? "lap" : "laps"}`;

  const circuits = Array.isArray(payload.circuits) ? payload.circuits : [];
  populateCircuitSelect(circuits, String(firstDefined(payload.default_circuit_id, "gb-1948")));
}

const VIEW_LABELS = {
  run: "Run simulation",
  results: "Results",
  history: "History"
};

function showView(viewName) {
  const name = Object.hasOwn(VIEW_LABELS, viewName) ? viewName : "run";
  for (const view of appViews) {
    const active = view.dataset.view === name;
    view.hidden = !active;
    view.classList.toggle("is-active", active);
  }
  for (const link of navigationLinks) {
    const active = link.dataset.viewTarget === name;
    link.classList.toggle("is-active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  currentViewLabel.textContent = VIEW_LABELS[name];
  if (name === "results") {
    requestAnimationFrame(() => {
      renderTrack(replayProgress);
      if (lapResults.classList.contains("is-visible")) {
        drawLapChart(chartLapTimes);
        drawSpeedTrace();
        if (tyreLapPoints.length) drawTyreWearChart(tyreLapPoints);
      }
    });
  }
  document.title = `${VIEW_LABELS[name]} | Vector Race`;
}

function navigateTo(viewName) {
  const hash = `#${viewName}`;
  if (window.location.hash === hash) showView(viewName);
  else window.location.hash = hash;
}

function viewFromHash() {
  const requested = window.location.hash.slice(1).toLowerCase();
  return Object.hasOwn(VIEW_LABELS, requested) ? requested : "run";
}

function normalizeHistoryEntry(raw, index = 0) {
  if (!raw || typeof raw !== "object") return null;
  const summary = raw.summary && typeof raw.summary === "object" ? raw.summary : raw;
  const rawCircuit = raw.circuit && typeof raw.circuit === "object" ? raw.circuit : {};
  const circuitId = String(firstDefined(raw.circuit_id, rawCircuit.id, raw.track_id, "")).trim();
  const catalogCircuit = circuitCatalog.find(circuit => circuit.id === circuitId);
  const circuitName = String(firstDefined(raw.circuit_name, raw.track_name, rawCircuit.name, catalogCircuit?.name, circuitId, "Unknown circuit"));
  const createdAt = String(firstDefined(raw.created_at, raw.timestamp, raw.run_at, raw.date, ""));
  const requestedLaps = Math.max(0, Math.round(finiteNumber(firstDefined(summary.requested_laps, raw.requested_laps, raw.laps), 0)));
  const completedLaps = Math.max(0, Math.round(finiteNumber(firstDefined(summary.completed_laps, raw.completed_laps), 0)));
  const explicitCompleted = firstDefined(summary.completed, raw.completed);
  return {
    id: String(firstDefined(raw.id, raw.history_id, `${createdAt}-${circuitId}-${index}`)),
    createdAt,
    circuitId,
    circuitName,
    location: String(firstDefined(raw.location, rawCircuit.location, catalogCircuit?.location, "")),
    countryCode: String(firstDefined(raw.country_code, rawCircuit.country_code, catalogCircuit?.countryCode, "")),
    requestedLaps,
    completedLaps,
    completed: typeof explicitCompleted === "boolean"
      ? explicitCompleted
      : requestedLaps > 0 && completedLaps >= requestedLaps,
    status: String(firstDefined(summary.termination_reason, summary.status, raw.termination_reason, raw.status, "")),
    totalTime: finiteNumber(firstDefined(summary.total_time_s, summary.simulated_time_s, raw.total_time_s), NaN),
    fastestLap: finiteNumber(firstDefined(summary.fastest_lap_s, raw.fastest_lap_s), NaN),
    averageLap: finiteNumber(firstDefined(summary.average_lap_s, raw.average_lap_s), NaN),
    maxSpeed: finiteNumber(firstDefined(summary.max_speed_kph, raw.max_speed_kph), NaN),
    offTrackEvents: Math.max(0, Math.round(finiteNumber(firstDefined(summary.off_track_events, raw.off_track_events), 0))),
    clearance: finiteNumber(firstDefined(
      summary.minimum_vehicle_edge_clearance_m,
      summary.clearance_m,
      raw.minimum_vehicle_edge_clearance_m,
      raw.clearance_m
    ), NaN),
    detailsAvailable: typeof firstDefined(raw.details_available, raw.has_details) === "boolean"
      ? Boolean(firstDefined(raw.details_available, raw.has_details))
      : null,
    strategy: raw.strategy && typeof raw.strategy === "object" ? raw.strategy : null,
    source: raw
  };
}

function historyDate(value) {
  if (!value) return "Unknown date";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(date);
}

function appendHistoryCell(row, label, value, className = "") {
  const cell = document.createElement("td");
  cell.dataset.label = label;
  cell.textContent = value;
  if (className) cell.className = className;
  row.append(cell);
}

function renderHistory() {
  historyTableBody.replaceChildren();
  historyNavCount.textContent = String(historyEntries.length);
  historyNavCount.setAttribute("aria-label", `${historyEntries.length} saved ${historyEntries.length === 1 ? "run" : "runs"}`);
  historyRunCount.textContent = String(historyEntries.length);
  const circuitKeys = new Set(historyEntries.map(entry => entry.circuitId || entry.circuitName));
  historyCircuitCount.textContent = String(circuitKeys.size);
  const fastest = historyEntries.map(entry => entry.fastestLap).filter(Number.isFinite);
  historyBestLap.textContent = fastest.length ? formatDuration(Math.min(...fastest)) : "\u2014";
  clearHistoryButton.disabled = historyLoading || Boolean(historyDetailLoadingId) || historyEntries.length === 0;

  if (!historyEntries.length) {
    historyPanel.hidden = true;
    historyMessage.hidden = false;
    historyMessage.textContent = historyLoading
      ? "Loading simulation history\u2026"
      : historyErrorMessage || "No simulations have been saved yet. Complete a run to start comparing circuit performance.";
    return;
  }

  historyMessage.hidden = true;
  historyPanel.hidden = false;
  for (const entry of historyEntries) {
    const row = document.createElement("tr");
    row.className = "history-result-row";
    row.dataset.historyId = entry.id;
    row.tabIndex = historyDetailLoadingId ? -1 : 0;
    row.setAttribute("aria-label", `View saved results for ${entry.circuitName}, ${historyDate(entry.createdAt)}`);
    if (historyDetailLoadingId === entry.id) row.setAttribute("aria-busy", "true");
    const circuitCell = document.createElement("td");
    circuitCell.dataset.label = "Circuit";
    const circuitName = document.createElement("strong");
    circuitName.textContent = entry.circuitName;
    const circuitLocation = document.createElement("span");
    circuitLocation.textContent = [entry.location, entry.countryCode].filter(Boolean).join(" \u00b7 ") || "F1 circuit";
    circuitCell.append(circuitName, circuitLocation);
    row.append(circuitCell);
    appendHistoryCell(row, "Run", historyDate(entry.createdAt));
    appendHistoryCell(row, "Laps", `${entry.completedLaps}/${entry.requestedLaps}`);
    appendHistoryCell(row, "Total time", formatDuration(entry.totalTime, true));
    appendHistoryCell(row, "Fastest lap", formatDuration(entry.fastestLap), Number.isFinite(entry.fastestLap) && entry.fastestLap === Math.min(...fastest) ? "history-best" : "");
    appendHistoryCell(row, "Average lap", formatDuration(entry.averageLap));
    appendHistoryCell(row, "Max speed", Number.isFinite(entry.maxSpeed) ? `${entry.maxSpeed.toFixed(1)} km/h` : "\u2014");
    appendHistoryCell(row, "Incidents", String(entry.offTrackEvents), entry.offTrackEvents > 0 ? "history-warning" : "history-clean");
    const actionCell = document.createElement("td");
    actionCell.dataset.label = "Results";
    actionCell.className = "history-action-cell";
    const viewButton = document.createElement("button");
    viewButton.className = "history-view-action";
    viewButton.type = "button";
    viewButton.dataset.historyId = entry.id;
    viewButton.disabled = Boolean(historyDetailLoadingId);
    viewButton.textContent = historyDetailLoadingId === entry.id
      ? "Loading\u2026"
      : entry.detailsAvailable === false ? "View summary" : "View results";
    viewButton.setAttribute("aria-label", `${viewButton.textContent} for ${entry.circuitName} from ${historyDate(entry.createdAt)}`);
    actionCell.append(viewButton);
    row.append(actionCell);
    historyTableBody.append(row);
  }
}

function historyCollection(payload) {
  if (Array.isArray(payload)) return payload;
  if (!payload || typeof payload !== "object") return [];
  if (Array.isArray(payload.history)) return payload.history;
  if (Array.isArray(payload.runs)) return payload.runs;
  if (Array.isArray(payload.items)) return payload.items;
  return [];
}

async function loadHistory({ quiet = false } = {}) {
  if (historyLoading) return;
  historyLoading = true;
  if (!quiet && !historyEntries.length) renderHistory();
  try {
    const response = await fetch("/api/history", { headers: { "Accept": "application/json" } });
    const payload = await response.json();
    if (!response.ok) throw new Error(typeof payload?.error === "string" ? payload.error : "Simulation history could not be loaded.");
    historyErrorMessage = "";
    historyEntries = historyCollection(payload)
      .map(normalizeHistoryEntry)
      .filter(Boolean)
      .sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt));
  } catch (error) {
    if (!quiet) historyErrorMessage = error instanceof Error ? error.message : "Simulation history could not be loaded.";
  } finally {
    historyLoading = false;
    renderHistory();
    if (simulationData) populateComparison(simulationData);
  }
}

async function clearHistory() {
  if (historyLoading || historyDetailLoadingId || !historyEntries.length) return;
  if (!window.confirm("Clear every saved simulation result? This cannot be undone.")) return;
  historyLoading = true;
  historyErrorMessage = "";
  renderHistory();
  try {
    const response = await fetch("/api/history", {
      method: "DELETE",
      headers: { "Accept": "application/json" }
    });
    if (!response.ok) {
      let detail = "Simulation history could not be cleared.";
      try {
        const payload = await response.json();
        if (typeof payload?.error === "string") detail = payload.error;
      } catch (error) {
        // The status code still provides a safe failure path for an empty response.
      }
      throw new Error(detail);
    }
    historyEntries = [];
  } catch (error) {
    historyErrorMessage = error instanceof Error ? error.message : "Simulation history could not be cleared.";
    showError(historyErrorMessage);
  } finally {
    historyLoading = false;
    renderHistory();
    if (simulationData) populateComparison(simulationData);
  }
}

function pairedSeries(xValue, yValue, numericExtras = {}, stringExtras = {}) {
  const rawX = Array.isArray(xValue) ? xValue : [];
  const rawY = Array.isArray(yValue) ? yValue : [];
  const length = Math.min(rawX.length, rawY.length);
  const result = { x: [], y: [] };
  const sourceIndices = [];

  for (const key of Object.keys(numericExtras)) result[key] = [];
  for (const key of Object.keys(stringExtras)) result[key] = [];

  for (let index = 0; index < length; index += 1) {
    const x = Number(rawX[index]);
    const y = Number(rawY[index]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    result.x.push(x);
    result.y.push(y);
    sourceIndices.push(index);
  }

  for (const [key, value] of Object.entries(numericExtras)) {
    const source = Array.isArray(value) ? value : [];
    result[key] = sourceIndices.map(index => {
      const number = Number(source[index]);
      return Number.isFinite(number) ? number : NaN;
    });
  }
  for (const [key, value] of Object.entries(stringExtras)) {
    const source = Array.isArray(value) ? value : [];
    result[key] = sourceIndices.map(index => typeof source[index] === "string" ? source[index] : "");
  }
  return result;
}

function normalizeSummary(rawSummary, strategy = null) {
  const summary = rawSummary && typeof rawSummary === "object" ? rawSummary : {};
  return {
    requestedLaps: Math.max(0, Math.round(finiteNumber(firstDefined(summary.requested_laps, summary.requestedLaps)))),
    completedLaps: Math.max(0, Math.round(finiteNumber(firstDefined(summary.completed_laps, summary.completedLaps)))),
    completed: typeof summary.completed === "boolean" ? summary.completed : null,
    lapTimes: numericArray(firstDefined(summary.lap_times_s, summary.lapTimes)),
    totalTime: finiteNumber(firstDefined(summary.total_time_s, summary.totalTime), NaN),
    fastestLap: finiteNumber(firstDefined(summary.fastest_lap_s, summary.fastestLap), NaN),
    averageLap: finiteNumber(firstDefined(summary.average_lap_s, summary.averageLap), NaN),
    offTrackEvents: Math.max(0, Math.round(finiteNumber(firstDefined(summary.off_track_events, summary.offTrackEvents)))),
    maxSpeed: finiteNumber(firstDefined(summary.max_speed_kph, summary.maxSpeed), NaN),
    meanSpeed: finiteNumber(firstDefined(summary.mean_speed_kph, summary.meanSpeed), NaN),
    clearance: finiteNumber(firstDefined(
      summary.minimum_vehicle_edge_clearance_m,
      summary.clearance_m,
      summary.clearance
    ), NaN),
    terminationReason: typeof firstDefined(summary.termination_reason, summary.terminationReason) === "string"
      ? firstDefined(summary.termination_reason, summary.terminationReason)
      : "Unknown",
    strategy: strategy || summary.strategy || null,
    pitStops: Math.max(0, Math.round(finiteNumber(firstDefined(summary.pit_stops, summary.pitStops)))),
    pitStopTime: Math.max(0, finiteNumber(firstDefined(summary.pit_stop_time_s, summary.pitStopTime)))
  };
}

function normalizeResponse(payload) {
  if (!payload || typeof payload !== "object") throw new Error("The server returned an empty response.");
  if (!payload.track || !payload.trajectory || !payload.telemetry || !payload.summary) {
    throw new Error("The simulation response is missing track, trajectory, telemetry, or summary data.");
  }

  const track = pairedSeries(payload.track.x_m, payload.track.y_m, {
    left: payload.track.width_left_m,
    right: payload.track.width_right_m
  });
  const trajectory = pairedSeries(payload.trajectory.x_m, payload.trajectory.y_m, {
    speed: payload.trajectory.speed_kph,
    distance: payload.trajectory.s_m
  });
  const telemetry = pairedSeries(payload.telemetry.x_m, payload.telemetry.y_m, {
    time: payload.telemetry.time_s,
    lap: payload.telemetry.lap_number,
    progress: payload.telemetry.progress_laps,
    speed: payload.telemetry.speed_kph,
    targetSpeed: payload.telemetry.target_speed_kph,
    clearance: payload.telemetry.clearance_m || payload.telemetry.vehicle_edge_clearance_m
  }, {
    tyre: payload.telemetry.tyre || payload.telemetry.tyre_compound,
    weather: payload.telemetry.weather || payload.telemetry.weather_condition
  });

  if (track.x.length < 3) throw new Error("The server did not return enough circuit points to draw the track.");
  if (trajectory.x.length < 2) throw new Error("The server did not return enough racing-line points.");

  const rawSummary = payload.summary;
  const selected = selectedCircuit();
  return {
    track: {
      id: String(firstDefined(payload.track.id, payload.track.circuit_id, selected?.id, "")),
      name: typeof payload.track.name === "string" ? payload.track.name : selected?.name || "F1 Circuit",
      location: String(firstDefined(payload.track.location, selected?.location, "")),
      countryCode: String(firstDefined(payload.track.country_code, selected?.countryCode, "")),
      lengthM: finiteNumber(firstDefined(payload.track.length_m, selected?.lengthM), NaN),
      ...track
    },
    trajectory,
    telemetry,
    summary: normalizeSummary(rawSummary, payload.strategy || rawSummary.strategy || null),
    tyreAnalysis: firstDefined(payload.tyre_analysis, rawSummary.tyre_analysis, null),
    history: payload.history && typeof payload.history === "object" ? payload.history : null
  };
}

function setLoading(isLoading) {
  for (const control of form.elements) control.disabled = isLoading;
  runButton.classList.toggle("is-loading", isLoading);
  runButtonText.textContent = isLoading ? "Running model\u2026" : "Run strategy";
  form.setAttribute("aria-busy", String(isLoading));
  if (isLoading) setSummarySkeleton(true);
}

function setSummarySkeleton(active) {
  const ids = ["completedLaps", "totalTime", "fastestLap", "averageLap", "maxSpeed", "meanSpeed", "clearance", "incidentCount"];
  for (const id of ids) document.getElementById(id).classList.toggle("skeleton", active);
}

function showError(message) {
  statusMessage.textContent = message;
  statusMessage.classList.add("is-visible");
  statusMessage.scrollIntoView({ behavior: reducedMotion.matches ? "auto" : "smooth", block: "nearest" });
}

function clearError() {
  statusMessage.textContent = "";
  statusMessage.classList.remove("is-visible");
}

function formatDuration(seconds, showHours = false) {
  if (!Number.isFinite(seconds) || seconds < 0) return "\u2014";
  const wholeMinutes = Math.floor(seconds / 60);
  const hours = Math.floor(wholeMinutes / 60);
  const minutes = wholeMinutes % 60;
  const secs = seconds % 60;
  const secondText = secs.toFixed(3).padStart(6, "0");
  if (showHours && hours > 0) return `${hours}:${String(minutes).padStart(2, "0")}:${secondText}`;
  return `${wholeMinutes}:${secondText}`;
}

function formatMetric(value, digits, unit) {
  if (!Number.isFinite(value)) return "\u2014";
  const fragment = document.createDocumentFragment();
  fragment.append(document.createTextNode(value.toFixed(digits) + " "));
  const unitNode = document.createElement("span");
  unitNode.className = "metric-unit";
  unitNode.textContent = unit;
  fragment.append(unitNode);
  return fragment;
}

function replaceContent(element, content) {
  element.replaceChildren();
  if (content instanceof Node) element.append(content);
  else element.textContent = String(content);
}

function isCompletedReason(reason) {
  const value = reason.trim().toLowerCase().replace(/[_-]+/g, " ");
  return value === "completed"
    || value === "complete"
    || value === "lap complete"
    || value === "requested laps completed"
    || value === "target laps completed"
    || value === "finished";
}

function readableReason(reason) {
  return reason.replace(/[_-]+/g, " ").trim() || "Unknown";
}

function readableToken(value) {
  return String(value || "").replace(/[_-]+/g, " ").replace(/\b\w/g, character => character.toUpperCase());
}

function populateSummary(summary) {
  setSummarySkeleton(false);
  document.getElementById("summaryTitle").textContent = summary.completedLaps > 0 ? "Simulation complete" : "Simulation stopped";
  document.getElementById("completedLaps").textContent = String(summary.completedLaps);
  document.getElementById("requestedLaps").textContent = `of ${summary.requestedLaps} ${summary.requestedLaps === 1 ? "lap" : "laps"}`;
  document.getElementById("summaryCaption").textContent = summary.completedLaps === summary.requestedLaps
    ? "Requested race distance completed."
    : "The model ended before the requested distance.";

  const successful = summary.completedLaps === summary.requestedLaps && isCompletedReason(summary.terminationReason);
  const badge = document.getElementById("terminationBadge");
  badge.classList.toggle("warning", !successful);
  document.getElementById("terminationText").textContent = readableReason(summary.terminationReason);

  document.getElementById("totalTime").textContent = formatDuration(summary.totalTime, true);
  document.getElementById("fastestLap").textContent = formatDuration(summary.fastestLap);
  document.getElementById("averageLap").textContent = formatDuration(summary.averageLap);
  replaceContent(document.getElementById("maxSpeed"), formatMetric(summary.maxSpeed, 1, "km/h"));
  replaceContent(document.getElementById("meanSpeed"), formatMetric(summary.meanSpeed, 1, "km/h"));
  replaceContent(document.getElementById("clearance"), formatMetric(summary.clearance, 2, "m"));

  const incidentCount = document.getElementById("incidentCount");
  incidentCount.textContent = String(summary.offTrackEvents);
  incidentCount.classList.toggle("has-events", summary.offTrackEvents > 0);
  document.getElementById("incidentLabel").textContent = summary.offTrackEvents === 1 ? "Off-track event" : "Off-track events";
}

const SEMANTIC_CLASSES = ["is-good", "is-bad", "is-neutral", "is-gain", "is-loss"];

function setSemanticState(element, state = "neutral") {
  if (!element) return;
  element.classList.remove(...SEMANTIC_CLASSES);
  element.classList.add(`is-${state}`);
}

function average(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : NaN;
}

function populateConsistency(lapTimes) {
  if (!consistencyRating || !consistencyStdDev || !consistencyRange || !consistencyTrend) return;
  const times = numericArray(lapTimes).filter(value => value >= 0);
  for (const element of [consistencyRating, consistencyStdDev, consistencyRange, consistencyTrend]) {
    setSemanticState(element, "neutral");
    element.removeAttribute("title");
  }

  if (times.length < 2) {
    consistencyRating.textContent = times.length ? "Need another lap" : "No completed laps";
    consistencyStdDev.textContent = "\u2014";
    consistencyRange.textContent = "\u2014";
    consistencyTrend.textContent = times.length
      ? "Complete at least two laps to measure a pace trend."
      : "A pace trend will appear after completed laps are returned.";
    return;
  }

  const mean = average(times);
  const variance = average(times.map(value => (value - mean) ** 2));
  const standardDeviation = Math.sqrt(Math.max(0, variance));
  const range = Math.max(...times) - Math.min(...times);
  const coefficientPercent = mean > 0 ? standardDeviation / mean * 100 : Infinity;
  let rating;
  let ratingState;
  if (coefficientPercent <= 0.35) {
    rating = "Excellent";
    ratingState = "good";
  } else if (coefficientPercent <= 0.75) {
    rating = "Very consistent";
    ratingState = "good";
  } else if (coefficientPercent <= 1.5) {
    rating = "Steady";
    ratingState = "neutral";
  } else if (coefficientPercent <= 3) {
    rating = "Variable";
    ratingState = "bad";
  } else {
    rating = "Highly variable";
    ratingState = "bad";
  }

  consistencyRating.textContent = rating;
  consistencyRating.title = `${coefficientPercent.toFixed(2)}% lap-time variation`;
  setSemanticState(consistencyRating, ratingState);
  consistencyStdDev.textContent = `${standardDeviation.toFixed(3)} s`;
  consistencyRange.textContent = `${range.toFixed(3)} s`;

  const split = Math.ceil(times.length / 2);
  const firstHalfAverage = average(times.slice(0, split));
  const secondHalfAverage = average(times.slice(split));
  const trend = secondHalfAverage - firstHalfAverage;
  const trendTolerance = 0.0005;
  if (Math.abs(trend) <= trendTolerance) {
    consistencyTrend.textContent = "Level pace between the first and second half (\u00b10.000 s).";
    setSemanticState(consistencyTrend, "neutral");
  } else if (trend < 0) {
    consistencyTrend.textContent = `${Math.abs(trend).toFixed(3)} s faster on average in the second half.`;
    setSemanticState(consistencyTrend, "good");
  } else {
    consistencyTrend.textContent = `${trend.toFixed(3)} s slower on average in the second half.`;
    setSemanticState(consistencyTrend, "bad");
  }
  consistencyTrend.title = `First half ${formatDuration(firstHalfAverage)}; second half ${formatDuration(secondHalfAverage)}.`;
}

function resultCompleted(summary) {
  if (!summary || summary.requestedLaps <= 0) return false;
  if (summary.completed === true) return summary.completedLaps >= summary.requestedLaps;
  if (summary.completed === false) return false;
  return summary.completedLaps >= summary.requestedLaps
    && (!summary.terminationReason || isCompletedReason(summary.terminationReason));
}

function signedDelta(delta, digits, unit = "") {
  const tolerance = 0.5 * (10 ** -digits);
  const effectivelyZero = Math.abs(delta) < tolerance;
  const sign = effectivelyZero ? "\u00b1" : delta > 0 ? "+" : "\u2212";
  const value = effectivelyZero ? 0 : Math.abs(delta);
  return `${sign}${value.toFixed(digits)}${unit ? ` ${unit}` : ""}`;
}

function setComparisonDelta(element, current, benchmark, options = {}) {
  if (!element) return;
  const {
    digits = 3,
    unit = "s",
    lowerIsBetter = true,
    comparable = true,
    unavailableReason = "This metric was not available for both runs."
  } = options;
  if (!comparable || !Number.isFinite(current) || !Number.isFinite(benchmark)) {
    element.textContent = "\u2014";
    element.title = unavailableReason;
    setSemanticState(element, "neutral");
    return;
  }

  const delta = current - benchmark;
  const tolerance = 0.5 * (10 ** -digits);
  const improvement = lowerIsBetter ? -delta : delta;
  element.textContent = signedDelta(delta, digits, unit);
  if (Math.abs(improvement) < tolerance) setSemanticState(element, "neutral");
  else setSemanticState(element, improvement > 0 ? "good" : "bad");
  element.title = `Current ${current.toFixed(digits)}${unit ? ` ${unit}` : ""}; benchmark ${benchmark.toFixed(digits)}${unit ? ` ${unit}` : ""}.`;
}

function resetComparison(message) {
  if (comparisonMessage) comparisonMessage.textContent = message;
  if (comparisonPanel) comparisonPanel.classList.remove("has-comparison");
  for (const element of Object.values(comparisonValues)) {
    if (!element) continue;
    element.textContent = "\u2014";
    element.title = "No prior same-circuit result is available.";
    setSemanticState(element, "neutral");
  }
}

function sameCircuitHistory(entry, data) {
  if (entry.circuitId && data.track.id) return entry.circuitId === data.track.id;
  return entry.circuitName.trim().toLowerCase() === data.track.name.trim().toLowerCase();
}

function comparisonBenchmark(data) {
  const currentHistory = normalizeHistoryEntry(data.history);
  const currentId = currentHistory?.id || "";
  const candidates = historyEntries.filter(entry => sameCircuitHistory(entry, data) && entry.id !== currentId);
  if (!candidates.length) return null;

  const sameDistance = candidates.filter(entry => entry.requestedLaps === data.summary.requestedLaps);
  const pool = sameDistance.length ? sameDistance : candidates;
  const timed = pool.filter(entry => Number.isFinite(entry.fastestLap));
  const entry = timed.length
    ? timed.reduce((best, candidate) => candidate.fastestLap < best.fastestLap ? candidate : best)
    : pool[0];
  return {
    entry,
    selectedByFastestLap: timed.length > 0,
    sameDistance: entry.requestedLaps === data.summary.requestedLaps
  };
}

function populateComparison(data) {
  if (!comparisonPanel || !comparisonTitle || !comparisonMessage) return;
  const benchmark = comparisonBenchmark(data);
  if (!benchmark) {
    comparisonTitle.textContent = `${data.track.name} benchmark`;
    resetComparison(`No earlier ${data.track.name} run is saved yet. Complete another run on this circuit to unlock a comparison.`);
    return;
  }

  const entry = benchmark.entry;
  const summary = data.summary;
  const fullDistanceComparable = benchmark.sameDistance
    && resultCompleted(summary)
    && entry.completed
    && summary.completedLaps === entry.completedLaps;
  comparisonPanel.classList.add("has-comparison");
  comparisonTitle.textContent = benchmark.selectedByFastestLap
    ? `Against best prior ${data.track.name} run`
    : `Against previous ${data.track.name} run`;

  const benchmarkDescription = `${entry.completedLaps}/${entry.requestedLaps} laps on ${historyDate(entry.createdAt)}`;
  if (!benchmark.sameDistance) {
    comparisonMessage.textContent = `Benchmark: ${benchmarkDescription}. Lap pace, speed, incidents and clearance remain comparable; total time is not because the race distances differ.`;
  } else if (!fullDistanceComparable) {
    comparisonMessage.textContent = `Benchmark: ${benchmarkDescription}. Total time is not compared because at least one run did not complete the full distance.`;
  } else {
    comparisonMessage.textContent = `Benchmark: ${benchmarkDescription}. Negative time and incident deltas, or positive speed and clearance deltas, are gains.`;
  }

  setComparisonDelta(comparisonValues.fastest, summary.fastestLap, entry.fastestLap);
  setComparisonDelta(comparisonValues.average, summary.averageLap, entry.averageLap);
  setComparisonDelta(comparisonValues.total, summary.totalTime, entry.totalTime, {
    comparable: fullDistanceComparable,
    unavailableReason: benchmark.sameDistance
      ? "Total time requires two completed runs over the full distance."
      : "Total time is not comparable across different race distances."
  });
  setComparisonDelta(comparisonValues.speed, summary.maxSpeed, entry.maxSpeed, {
    digits: 1,
    unit: "km/h",
    lowerIsBetter: false
  });
  setComparisonDelta(comparisonValues.incidents, summary.offTrackEvents, entry.offTrackEvents, {
    digits: 0,
    unit: "",
    lowerIsBetter: true
  });
  setComparisonDelta(comparisonValues.clearance, summary.clearance, entry.clearance, {
    digits: 2,
    unit: "m",
    lowerIsBetter: false
  });
}

function planItemForLap(items, lap) {
  if (!Array.isArray(items) || !items.length) return null;
  let active = null;
  for (const item of items) {
    if (!item || !Number.isFinite(Number(item.start_lap))) continue;
    if (Number(item.start_lap) <= lap) active = item;
    else break;
  }
  return active;
}

function lapConditionCell(kind, token) {
  const cell = document.createElement("td");
  cell.dataset.label = kind === "tyre" ? "Tyre" : "Weather";
  const catalogue = kind === "tyre" ? TYRES : WEATHER;
  const item = catalogue[token];
  if (!token) {
    cell.textContent = "\u2014";
    return cell;
  }
  const badge = document.createElement("span");
  const safeToken = String(token).toLowerCase().replace(/[^a-z0-9_-]+/g, "-");
  badge.className = `lap-condition-badge ${kind}-badge ${kind}-${safeToken}`;
  badge.style.setProperty("--badge-colour", item?.colour || "#9ea9bc");
  badge.textContent = item?.label || readableToken(token);
  badge.title = kind === "tyre" ? `${badge.textContent} tyre` : `${badge.textContent} conditions`;
  cell.append(badge);
  return cell;
}

function populateLapTable(lapTimes, plan = null) {
  const body = document.getElementById("lapTableBody");
  body.replaceChildren();
  if (!lapTimes.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.textContent = "No completed lap times were returned.";
    row.append(cell);
    body.append(row);
    return;
  }

  const fastest = Math.min(...lapTimes);
  lapTimes.forEach((time, index) => {
    const row = document.createElement("tr");
    const isFastest = Math.abs(time - fastest) < 0.0005;
    if (isFastest) row.className = "fastest";

    const lapCell = document.createElement("td");
    lapCell.dataset.label = "Lap";
    lapCell.textContent = `Lap ${index + 1}`;
    const lapNumber = index + 1;
    const tyre = planItemForLap(plan?.stints, lapNumber)?.tyre || "";
    const weather = planItemForLap(plan?.weather, lapNumber)?.condition || "";
    const timeCell = document.createElement("td");
    timeCell.dataset.label = "Time";
    timeCell.textContent = formatDuration(time);
    const deltaCell = document.createElement("td");
    deltaCell.dataset.label = "Delta";
    const delta = time - fastest;
    deltaCell.textContent = isFastest ? "FASTEST" : `+${delta.toFixed(3)} s`;
    deltaCell.className = isFastest ? "delta-zero" : "delta-positive";
    row.append(lapCell, lapConditionCell("tyre", tyre), lapConditionCell("weather", weather), timeCell, deltaCell);
    body.append(row);
  });
}

function appendOptions(select, entries, selectedValue) {
  for (const [value, label] of entries) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    option.selected = value === selectedValue;
    select.append(option);
  }
}

function createField(labelText, control, extraClass = "") {
  const label = document.createElement("label");
  label.className = `field-group ${extraClass}`.trim();
  const labelSpan = document.createElement("span");
  labelSpan.textContent = labelText;
  label.append(labelSpan, control);
  return label;
}

function createNumberInput(className, value, min, max, step, accessibleLabel) {
  const input = document.createElement("input");
  input.className = className;
  input.type = "number";
  input.value = String(value);
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.inputMode = "decimal";
  input.setAttribute("aria-label", accessibleLabel);
  return input;
}

function createRemoveButton(label) {
  const button = document.createElement("button");
  button.className = "remove-row-button";
  button.type = "button";
  button.textContent = "x";
  button.setAttribute("aria-label", label);
  button.title = label;
  button.addEventListener("click", () => {
    button.closest(".strategy-row").remove();
    renderStrategyTimeline();
  });
  return button;
}

function createPitStopRow(startLap, tyre = "intermediate") {
  rowSequence += 1;
  const row = document.createElement("div");
  row.className = "strategy-row pit-row";
  row.dataset.pitRow = String(rowSequence);

  const lapInput = createNumberInput("pit-start", startLap, 2, maximumLaps, 1, "New tyre starts on lap");
  const tyreSelect = document.createElement("select");
  tyreSelect.className = "pit-tyre";
  tyreSelect.setAttribute("aria-label", "Tyre after pit stop");
  appendOptions(tyreSelect, Object.entries(TYRES).map(([value, item]) => [value, item.label]), tyre);

  row.append(
    createField("New tyre from lap", lapInput, "grow"),
    createField("Compound", tyreSelect, "grow"),
    createRemoveButton("Remove pit stop")
  );
  pitStopRows.append(row);
  renderStrategyTimeline();
  return row;
}

function createWeatherRow(startLap, condition, temperature, rainPercent, removable = true) {
  rowSequence += 1;
  const row = document.createElement("div");
  row.className = "strategy-row weather-row";
  row.dataset.weatherRow = String(rowSequence);

  if (removable) {
    const lapInput = createNumberInput("weather-start", startLap, 2, maximumLaps, 1, "Weather phase starts on lap");
    row.append(createField("From lap", lapInput));
  } else {
    row.dataset.startLap = "1";
    const lapChip = document.createElement("span");
    lapChip.className = "lap-chip";
    lapChip.textContent = "Lap 1";
    row.append(lapChip);
  }

  const conditionSelect = document.createElement("select");
  conditionSelect.className = "weather-condition";
  conditionSelect.setAttribute("aria-label", "Track condition");
  appendOptions(conditionSelect, Object.entries(WEATHER).map(([value, item]) => [value, item.label]), condition);
  const temperatureInput = createNumberInput("track-temperature", temperature, -10, 70, 1, "Track temperature in degrees Celsius");
  const rainInput = createNumberInput("rain-intensity", rainPercent, 0, 100, 1, "Rain intensity percentage");

  row.append(
    createField("Condition", conditionSelect, "condition-field"),
    createField("Track temp (C)", temperatureInput, "weather-number"),
    createField("Rain (%)", rainInput, "weather-number")
  );
  if (removable) row.append(createRemoveButton("Remove weather change"));
  weatherRows.append(row);
  renderStrategyTimeline();
  return row;
}

function currentLapCount() {
  const laps = Number(lapsInput.value);
  return Number.isInteger(laps) && laps >= 1 && laps <= maximumLaps ? laps : defaultLaps;
}

function occupiedStarts(selector) {
  return new Set([...document.querySelectorAll(selector)].map(input => Number(input.value)).filter(Number.isInteger));
}

function availableStart(selector, preferred) {
  const laps = currentLapCount();
  const used = occupiedStarts(selector);
  for (let lap = Math.max(2, Math.min(laps, preferred)); lap <= laps; lap += 1) {
    if (!used.has(lap)) return lap;
  }
  for (let lap = 2; lap <= laps; lap += 1) {
    if (!used.has(lap)) return lap;
  }
  return null;
}

function strategyDraft(clampValues = true) {
  const laps = currentLapCount();
  const stints = [{ start_lap: 1, tyre: TYRES[initialTyre.value] ? initialTyre.value : "medium" }];
  for (const row of pitStopRows.querySelectorAll(".pit-row")) {
    const parsedStart = finiteNumber(row.querySelector(".pit-start").value, NaN);
    const rawStart = clampValues ? Math.round(parsedStart) : parsedStart;
    const startLap = clampValues ? Math.max(2, Math.min(laps, Number.isFinite(rawStart) ? rawStart : 2)) : rawStart;
    const tyre = row.querySelector(".pit-tyre").value;
    stints.push({ start_lap: startLap, tyre: TYRES[tyre] ? tyre : "medium" });
  }
  stints.sort((a, b) => a.start_lap - b.start_lap);

  const weather = [];
  for (const row of weatherRows.querySelectorAll(".weather-row")) {
    const startInput = row.querySelector(".weather-start");
    const parsedStart = startInput ? finiteNumber(startInput.value, NaN) : 1;
    const rawStart = clampValues ? Math.round(parsedStart) : parsedStart;
    const startLap = clampValues ? Math.max(startInput ? 2 : 1, Math.min(laps, Number.isFinite(rawStart) ? rawStart : 2)) : rawStart;
    const condition = row.querySelector(".weather-condition").value;
    weather.push({
      start_lap: startLap,
      condition: WEATHER[condition] ? condition : "dry",
      track_temp_c: finiteNumber(row.querySelector(".track-temperature").value, 25),
      rain_intensity: finiteNumber(row.querySelector(".rain-intensity").value, 0) / 100
    });
  }
  weather.sort((a, b) => a.start_lap - b.start_lap);
  return { laps, stints, weather };
}

function validateUniqueStarts(items, label) {
  const starts = new Set();
  for (const item of items) {
    if (starts.has(item.start_lap)) throw new Error(`${label} cannot contain two changes on lap ${item.start_lap}.`);
    starts.add(item.start_lap);
  }
}

function collectPlan() {
  if (!circuitSelect.value || !circuitCatalog.some(circuit => circuit.id === circuitSelect.value)) {
    throw new Error("Choose a valid Formula 1 circuit.");
  }
  const laps = Number(lapsInput.value);
  if (!Number.isInteger(laps) || laps < 1 || laps > maximumLaps) {
    throw new Error(`Enter a whole number from 1 to ${maximumLaps} laps.`);
  }

  const plan = strategyDraft(false);
  for (const stint of plan.stints) {
    if (!TYRES[stint.tyre]) throw new Error("Choose a valid tyre for every stint.");
    if (!Number.isInteger(stint.start_lap) || stint.start_lap < 1 || stint.start_lap > laps) {
      throw new Error(`Every tyre stint must start between laps 1 and ${laps}.`);
    }
  }
  validateUniqueStarts(plan.stints, "The tyre plan");
  for (let index = 1; index < plan.stints.length; index += 1) {
    if (plan.stints[index - 1].tyre === plan.stints[index].tyre) {
      throw new Error(`Choose a different tyre for the stint starting on lap ${plan.stints[index].start_lap}, or remove that pit stop.`);
    }
  }

  for (const phase of plan.weather) {
    if (!WEATHER[phase.condition]) throw new Error("Choose a valid condition for every weather phase.");
    if (!Number.isInteger(phase.start_lap) || phase.start_lap < 1 || phase.start_lap > laps) {
      throw new Error(`Every weather phase must start between laps 1 and ${laps}.`);
    }
    if (!Number.isFinite(phase.track_temp_c) || phase.track_temp_c < -10 || phase.track_temp_c > 70) {
      throw new Error("Track temperature must be between -10 and 70 C.");
    }
    if (!Number.isFinite(phase.rain_intensity) || phase.rain_intensity < 0 || phase.rain_intensity > 1) {
      throw new Error("Rain intensity must be between 0 and 100 percent.");
    }
  }
  validateUniqueStarts(plan.weather, "The weather plan");
  return plan;
}

function activePhase(items, lap) {
  let active = items[0];
  for (const item of items) {
    if (item.start_lap <= lap) active = item;
    else break;
  }
  return active;
}

function renderStrategyTimeline() {
  const plan = strategyDraft();
  strategyStatus.textContent = `${plan.laps}-lap plan \u00b7 ${Math.max(0, plan.stints.length - 1)} ${plan.stints.length === 2 ? "stop" : "stops"}`;
  strategyTimeline.replaceChildren();
  const track = document.createElement("div");
  track.className = "timeline-track";
  track.style.setProperty("--lap-count", String(plan.laps));
  const pitLaps = new Set(plan.stints.slice(1).map(stint => stint.start_lap));
  const weatherChanges = new Set(plan.weather.slice(1).map(phase => phase.start_lap));

  for (let lap = 1; lap <= plan.laps; lap += 1) {
    const stint = activePhase(plan.stints, lap);
    const phase = activePhase(plan.weather, lap);
    const cell = document.createElement("div");
    cell.className = `timeline-lap tyre-${stint.tyre} weather-${phase.condition}`;
    cell.classList.toggle("is-pit", pitLaps.has(lap));
    cell.classList.toggle("is-change", weatherChanges.has(lap));
    const lapLabel = document.createElement("strong");
    lapLabel.textContent = `L${lap}`;
    const tyreLabel = document.createElement("span");
    tyreLabel.textContent = TYRES[stint.tyre].short;
    cell.title = `Lap ${lap}: ${TYRES[stint.tyre].label}; ${WEATHER[phase.condition].label}, ${phase.track_temp_c.toFixed(0)} C, ${(phase.rain_intensity * 100).toFixed(0)}% rain${pitLaps.has(lap) ? "; pit stop" : ""}${weatherChanges.has(lap) ? "; weather change" : ""}.`;
    cell.append(lapLabel, tyreLabel);
    track.append(cell);
  }
  strategyTimeline.append(track);
}

function populateExecutedPlan(plan, summary) {
  const container = document.getElementById("executedPlanBody");
  container.replaceChildren();
  const stopCount = Number.isFinite(summary.pitStops) ? summary.pitStops : Math.max(0, plan.stints.length - 1);
  const pitLoss = summary.pitStopTime > 0 ? ` \u00b7 ${summary.pitStopTime.toFixed(1)} s` : "";
  document.getElementById("pitCount").textContent = `${stopCount} ${stopCount === 1 ? "stop" : "stops"}${pitLoss}`;

  const sections = [
    { title: "Tyre stints", items: plan.stints, kind: "tyre" },
    { title: "Weather phases", items: plan.weather, kind: "weather" }
  ];
  for (const sectionData of sections) {
    const section = document.createElement("section");
    section.className = "plan-section";
    const heading = document.createElement("h3");
    heading.className = "plan-section-title";
    heading.textContent = sectionData.title;
    const cards = document.createElement("div");
    cards.className = "plan-cards";

    sectionData.items.forEach((item, index) => {
      const next = sectionData.items[index + 1];
      const endLap = next ? next.start_lap - 1 : plan.laps;
      const card = document.createElement("div");
      card.className = sectionData.kind === "tyre" ? `plan-card tyre-${item.tyre}` : `plan-card weather-${item.condition}`;
      const title = document.createElement("strong");
      const detail = document.createElement("span");
      if (sectionData.kind === "tyre") {
        title.textContent = TYRES[item.tyre].label;
        detail.textContent = `${item.start_lap === endLap ? "Lap" : "Laps"} ${item.start_lap}${item.start_lap === endLap ? "" : `\u2013${endLap}`}${index > 0 ? " \u00b7 pit exit" : " \u00b7 race start"}`;
      } else {
        title.textContent = WEATHER[item.condition].label;
        detail.textContent = `Laps ${item.start_lap}${item.start_lap === endLap ? "" : `\u2013${endLap}`} \u00b7 ${item.track_temp_c.toFixed(0)} C \u00b7 ${(item.rain_intensity * 100).toFixed(0)}% rain`;
      }
      card.append(title, detail);
      cards.append(card);
    });
    section.append(heading, cards);
    container.append(section);
  }
}

function initializeStrategy() {
  applyPreset("dry");
}

function applyPreset(name) {
  const laps = currentLapCount();
  pitStopRows.replaceChildren();
  weatherRows.replaceChildren();
  initialTyre.value = "medium";

  if (name === "rain") {
    createWeatherRow(1, "dry", 25, 0, false);
    for (const [startLap, tyre] of [[4, "intermediate"], [7, "wet"]]) {
      if (startLap <= laps) createPitStopRow(startLap, tyre);
    }
    for (const [startLap, condition] of [[3, "light_rain"], [6, "wet"], [8, "heavy_rain"]]) {
      if (startLap <= laps) {
        const preset = WEATHER[condition];
        createWeatherRow(startLap, condition, preset.temperature, preset.rain, true);
      }
    }
  } else {
    createWeatherRow(1, "dry", 30, 0, false);
    if (laps > 1) createPitStopRow(Math.min(6, laps), "soft");
  }
  renderStrategyTimeline();
}

function fitCanvas(canvas, context) {
  const rect = canvas.getBoundingClientRect();
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 3);
  const width = Math.max(1, Math.round(rect.width * pixelRatio));
  const height = Math.max(1, Math.round(rect.height * pixelRatio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  return { width: rect.width, height: rect.height, pixelRatio };
}

function clampMapPan(width, height) {
  if (mapView.zoom <= MIN_ZOOM + 0.001) {
    mapView.panX = 0;
    mapView.panY = 0;
    return;
  }
  const maximumX = width * ((mapView.zoom - 1) * 0.52 + 0.12);
  const maximumY = height * ((mapView.zoom - 1) * 0.52 + 0.12);
  mapView.panX = Math.max(-maximumX, Math.min(maximumX, mapView.panX));
  mapView.panY = Math.max(-maximumY, Math.min(maximumY, mapView.panY));
}

function trackGeometry(data, width, height) {
  const xs = data.track.x;
  const ys = data.track.y;
  const finiteWidths = [...data.track.left, ...data.track.right].filter(Number.isFinite);
  const maxWidth = finiteWidths.length ? Math.max(1, ...finiteWidths) : 8;
  const minX = Math.min(...xs) - maxWidth;
  const maxX = Math.max(...xs) + maxWidth;
  const minY = Math.min(...ys) - maxWidth;
  const maxY = Math.max(...ys) + maxWidth;
  const rangeX = Math.max(1, maxX - minX);
  const rangeY = Math.max(1, maxY - minY);
  const compact = width < 620;
  const paddingX = compact ? 32 : 76;
  const paddingTop = compact ? 118 : 74;
  const paddingBottom = compact ? 100 : 74;
  const availableWidth = Math.max(1, width - paddingX * 2);
  const availableHeight = Math.max(1, height - paddingTop - paddingBottom);
  const baseScale = Math.min(availableWidth / rangeX, availableHeight / rangeY);
  const drawnWidth = rangeX * baseScale;
  const drawnHeight = rangeY * baseScale;
  const originX = (width - drawnWidth) / 2;
  const originY = paddingTop + (availableHeight - drawnHeight) / 2;
  const centreX = width / 2;
  const centreY = height / 2;
  clampMapPan(width, height);
  return {
    point(x, y) {
      const baseX = originX + (x - minX) * baseScale;
      const baseY = originY + drawnHeight - (y - minY) * baseScale;
      return {
        x: centreX + (baseX - centreX) * mapView.zoom + mapView.panX,
        y: centreY + (baseY - centreY) * mapView.zoom + mapView.panY
      };
    },
    scale: baseScale * mapView.zoom,
    width,
    height
  };
}

function drawBackgroundGrid(context, width, height) {
  context.save();
  context.strokeStyle = "rgba(255, 255, 255, 0.025)";
  context.lineWidth = 1;
  const spacing = Math.max(28, Math.min(90, 36 * Math.sqrt(mapView.zoom)));
  const offsetX = ((mapView.panX % spacing) + spacing) % spacing;
  const offsetY = ((mapView.panY % spacing) + spacing) % spacing;
  context.beginPath();
  for (let x = offsetX + 0.5; x < width; x += spacing) { context.moveTo(x, 0); context.lineTo(x, height); }
  for (let y = offsetY + 0.5; y < height; y += spacing) { context.moveTo(0, y); context.lineTo(width, y); }
  context.stroke();
  context.restore();
}

function widthsAt(track, index) {
  const fallback = 7;
  return {
    left: Number.isFinite(track.left[index]) ? Math.max(0, track.left[index]) : fallback,
    right: Number.isFinite(track.right[index]) ? Math.max(0, track.right[index]) : fallback
  };
}

function offsetTrackEdges(track) {
  const left = [];
  const right = [];
  const count = track.x.length;
  const closed = count > 3 && Math.hypot(track.x[0] - track.x[count - 1], track.y[0] - track.y[count - 1]) < 30;
  for (let index = 0; index < count; index += 1) {
    const previous = closed ? (index - 1 + count) % count : Math.max(0, index - 1);
    const next = closed ? (index + 1) % count : Math.min(count - 1, index + 1);
    const dx = track.x[next] - track.x[previous];
    const dy = track.y[next] - track.y[previous];
    const length = Math.hypot(dx, dy) || 1;
    const nx = -dy / length;
    const ny = dx / length;
    const widths = widthsAt(track, index);
    left.push({ x: track.x[index] + nx * widths.left, y: track.y[index] + ny * widths.left });
    right.push({ x: track.x[index] - nx * widths.right, y: track.y[index] - ny * widths.right });
  }
  return { left, right, closed };
}

function drawPolyline(context, points, close = false) {
  if (!points.length) return;
  context.beginPath();
  context.moveTo(points[0].x, points[0].y);
  for (let index = 1; index < points.length; index += 1) context.lineTo(points[index].x, points[index].y);
  if (close) context.closePath();
}

function drawTrack(context, data, geometry) {
  const edges = offsetTrackEdges(data.track);
  const left = edges.left.map(point => geometry.point(point.x, point.y));
  const right = edges.right.map(point => geometry.point(point.x, point.y));
  const center = data.track.x.map((x, index) => geometry.point(x, data.track.y[index]));

  context.save();
  context.shadowColor = "rgba(0, 0, 0, 0.45)";
  context.shadowBlur = 22;
  drawPolyline(context, [...left, ...right.slice().reverse()], true);
  context.fillStyle = "#232a36";
  context.fill();
  context.restore();

  context.save();
  context.lineJoin = "round";
  context.lineCap = "round";
  context.lineWidth = Math.min(2.2, Math.max(1.25, geometry.scale * 0.5));
  context.strokeStyle = "rgba(181, 193, 211, 0.72)";
  drawPolyline(context, left, edges.closed);
  context.stroke();
  drawPolyline(context, right, edges.closed);
  context.stroke();
  context.setLineDash([3, 7]);
  context.lineWidth = 0.8;
  context.strokeStyle = "rgba(255, 255, 255, 0.12)";
  drawPolyline(context, center, edges.closed);
  context.stroke();
  context.restore();

  drawStartLine(context, data, geometry);
}

function drawStartLine(context, data, geometry) {
  if (data.track.x.length < 2) return;
  const first = geometry.point(data.track.x[0], data.track.y[0]);
  const second = geometry.point(data.track.x[1], data.track.y[1]);
  const angle = Math.atan2(second.y - first.y, second.x - first.x) + Math.PI / 2;
  const widths = widthsAt(data.track, 0);
  const halfLength = (widths.left + widths.right) * geometry.scale * 0.48;
  context.save();
  context.translate(first.x, first.y);
  context.rotate(angle);
  context.strokeStyle = "rgba(255, 255, 255, 0.9)";
  context.lineWidth = 2;
  context.setLineDash([3, 3]);
  context.beginPath();
  context.moveTo(-halfLength, 0);
  context.lineTo(halfLength, 0);
  context.stroke();
  context.restore();
}

function speedColour(value, minimum, maximum) {
  const ratio = maximum > minimum ? Math.max(0, Math.min(1, (value - minimum) / (maximum - minimum))) : 0.5;
  const hue = 210 - ratio * 205;
  return `hsl(${hue} 92% ${58 + ratio * 5}%)`;
}

function finiteExtent(values, fallbackMinimum = 0, fallbackMaximum = 1) {
  const finite = values.filter(Number.isFinite);
  return finite.length ? [Math.min(...finite), Math.max(...finite)] : [fallbackMinimum, fallbackMaximum];
}

function drawTrajectory(context, data, geometry) {
  const count = data.trajectory.x.length;
  if (count < 2) return;
  const heatmap = layerInputs.heatmap.checked;
  const [minimum, maximum] = finiteExtent(data.trajectory.speed);
  context.save();
  context.lineWidth = Math.max(1.8, Math.min(4.6, geometry.scale * 1.9));
  context.lineCap = "round";
  context.lineJoin = "round";
  context.shadowColor = heatmap ? "rgba(255, 176, 47, 0.24)" : "rgba(255, 209, 102, 0.34)";
  context.shadowBlur = 7;
  if (!heatmap) {
    const points = data.trajectory.x.map((x, index) => geometry.point(x, data.trajectory.y[index]));
    context.strokeStyle = "rgba(255, 209, 102, 0.96)";
    drawPolyline(context, points, false);
    context.stroke();
  } else {
    for (let index = 1; index < count; index += 1) {
      const previous = geometry.point(data.trajectory.x[index - 1], data.trajectory.y[index - 1]);
      const current = geometry.point(data.trajectory.x[index], data.trajectory.y[index]);
      context.strokeStyle = speedColour(data.trajectory.speed[index], minimum, maximum);
      context.beginPath();
      context.moveTo(previous.x, previous.y);
      context.lineTo(current.x, current.y);
      context.stroke();
    }
  }
  context.restore();
}

function drawDrivenPath(context, data, geometry, progress) {
  const count = data.telemetry.x.length;
  if (count < 1) return null;
  const lastFloat = Math.max(0, Math.min(count - 1, progress * (count - 1)));
  const last = Math.floor(lastFloat);
  const fraction = lastFloat - last;
  const step = Math.max(1, Math.floor(count / 3000));
  const heatmap = layerInputs.heatmap.checked && data.telemetry.speed.some(Number.isFinite);

  context.save();
  context.lineWidth = Math.max(1.5, Math.min(3.7, geometry.scale * 1.45));
  context.lineJoin = "round";
  context.lineCap = "round";
  context.shadowColor = "rgba(51, 214, 208, 0.48)";
  context.shadowBlur = 5;
  if (heatmap) {
    const [minimum, maximum] = finiteExtent(data.telemetry.speed);
    let previous = geometry.point(data.telemetry.x[0], data.telemetry.y[0]);
    for (let index = step; index <= last; index += step) {
      const point = geometry.point(data.telemetry.x[index], data.telemetry.y[index]);
      context.strokeStyle = speedColour(data.telemetry.speed[index], minimum, maximum);
      context.beginPath();
      context.moveTo(previous.x, previous.y);
      context.lineTo(point.x, point.y);
      context.stroke();
      previous = point;
    }
  } else {
    context.strokeStyle = "rgba(51, 214, 208, 0.9)";
    context.beginPath();
    const first = geometry.point(data.telemetry.x[0], data.telemetry.y[0]);
    context.moveTo(first.x, first.y);
    for (let index = step; index <= last; index += step) {
      const point = geometry.point(data.telemetry.x[index], data.telemetry.y[index]);
      context.lineTo(point.x, point.y);
    }
    if (last > 0 && last % step !== 0) {
      const point = geometry.point(data.telemetry.x[last], data.telemetry.y[last]);
      context.lineTo(point.x, point.y);
    }
    context.stroke();
  }
  context.restore();

  let worldX = data.telemetry.x[last];
  let worldY = data.telemetry.y[last];
  if (last < count - 1) {
    worldX += (data.telemetry.x[last + 1] - worldX) * fraction;
    worldY += (data.telemetry.y[last + 1] - worldY) * fraction;
  }
  return { point: geometry.point(worldX, worldY), index: last };
}

function drawCar(context, data, geometry, position) {
  if (!position) return;
  const count = data.telemetry.x.length;
  const behind = Math.max(0, position.index - 2);
  const ahead = Math.min(count - 1, position.index + 2);
  const first = geometry.point(data.telemetry.x[behind], data.telemetry.y[behind]);
  const second = geometry.point(data.telemetry.x[ahead], data.telemetry.y[ahead]);
  const angle = Math.atan2(second.y - first.y, second.x - first.x);
  const size = Math.max(7, Math.min(12, geometry.scale * 6));

  context.save();
  context.translate(position.point.x, position.point.y);
  context.rotate(angle);
  context.shadowColor = "rgba(51, 214, 208, 0.8)";
  context.shadowBlur = 15;
  context.fillStyle = "#ecffff";
  context.beginPath();
  context.moveTo(size, 0);
  context.lineTo(-size * 0.7, size * 0.54);
  context.lineTo(-size * 0.42, 0);
  context.lineTo(-size * 0.7, -size * 0.54);
  context.closePath();
  context.fill();
  context.fillStyle = "#16aaa5";
  context.beginPath();
  context.arc(size * 0.05, 0, size * 0.23, 0, Math.PI * 2);
  context.fill();
  context.restore();
}

function drawTelemetryHighlight(context, geometry) {
  if (!simulationData || hoveredTelemetryIndex < 0) return;
  const telemetry = simulationData.telemetry;
  if (hoveredTelemetryIndex >= telemetry.x.length) return;
  const point = geometry.point(telemetry.x[hoveredTelemetryIndex], telemetry.y[hoveredTelemetryIndex]);
  context.save();
  context.fillStyle = "#ffffff";
  context.strokeStyle = "rgba(51, 214, 208, 0.7)";
  context.lineWidth = 3;
  context.shadowColor = "rgba(51, 214, 208, 0.85)";
  context.shadowBlur = 12;
  context.beginPath();
  context.arc(point.x, point.y, 5, 0, Math.PI * 2);
  context.fill();
  context.stroke();
  context.restore();
}

function renderTrack(progress = replayProgress) {
  const dimensions = fitCanvas(trackCanvas, trackContext);
  trackContext.clearRect(0, 0, dimensions.width, dimensions.height);
  drawBackgroundGrid(trackContext, dimensions.width, dimensions.height);
  zoomLevel.textContent = `${Math.round(mapView.zoom * 100)}%`;
  if (!simulationData) return;
  const geometry = trackGeometry(simulationData, dimensions.width, dimensions.height);
  lastGeometry = geometry;
  if (layerInputs.track.checked) drawTrack(trackContext, simulationData, geometry);
  if (layerInputs.optimized.checked) drawTrajectory(trackContext, simulationData, geometry);
  const position = layerInputs.driven.checked ? drawDrivenPath(trackContext, simulationData, geometry, progress) : null;
  if (layerInputs.driven.checked) drawCar(trackContext, simulationData, geometry, position);
  drawTelemetryHighlight(trackContext, geometry);
  updateReplayReadout(position ? position.index : Math.floor(progress * Math.max(0, simulationData.telemetry.x.length - 1)));
}

function updateReplayReadout(index) {
  if (!simulationData || !simulationData.telemetry.x.length) {
    replayPosition.textContent = "No telemetry returned";
    return;
  }
  const telemetry = simulationData.telemetry;
  const bounded = Math.max(0, Math.min(index, telemetry.x.length - 1));
  const lap = Number.isFinite(telemetry.lap[bounded]) ? Math.max(1, Math.round(telemetry.lap[bounded])) : 1;
  const time = Number.isFinite(telemetry.time[bounded]) ? telemetry.time[bounded] : 0;
  const speed = Number.isFinite(telemetry.speed[bounded]) ? ` \u00b7 ${telemetry.speed[bounded].toFixed(0)} km/h` : "";
  replayPosition.textContent = `Lap ${lap} \u00b7 ${formatDuration(time, true)}${speed}`;
}

function replayDuration() {
  if (!simulationData) return 10000;
  const laps = Math.max(1, simulationData.summary.completedLaps);
  return Math.min(20000, Math.max(8000, laps * 1200));
}

function animateReplay(timestamp) {
  if (!replayRunning) return;
  if (!replayStartedAt) replayStartedAt = timestamp;
  replayProgress = Math.min(1, (timestamp - replayStartedAt) / replayDuration());
  renderTrack(replayProgress);
  if (replayProgress < 1) {
    replayFrame = requestAnimationFrame(animateReplay);
  } else {
    replayRunning = false;
    liveText.textContent = "Replay complete";
    trackCanvas.setAttribute("aria-label", `${simulationData.track.name} circuit map showing the optimized racing line and complete simulated driven path. Use the arrow keys to pan and plus or minus to zoom.`);
  }
}

function startReplay() {
  cancelAnimationFrame(replayFrame);
  replayProgress = reducedMotion.matches ? 1 : 0;
  replayStartedAt = 0;
  replayRunning = !reducedMotion.matches && simulationData && simulationData.telemetry.x.length > 1;
  liveText.textContent = replayRunning ? "Live replay" : "Replay complete";
  renderTrack(replayProgress);
  if (replayRunning) replayFrame = requestAnimationFrame(animateReplay);
}

function resetMapView() {
  mapView.zoom = 1;
  mapView.panX = 0;
  mapView.panY = 0;
  tooltipPinned = false;
  hoveredTelemetryIndex = -1;
  mapTooltip.hidden = true;
  renderTrack(replayProgress);
}

function zoomAt(newZoom, point, shouldRender = true) {
  const rect = trackCanvas.getBoundingClientRect();
  const previous = mapView.zoom;
  const bounded = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newZoom));
  if (Math.abs(bounded - previous) < 0.0001) return;
  const centreX = rect.width / 2;
  const centreY = rect.height / 2;
  const anchorX = point ? point.x : centreX;
  const anchorY = point ? point.y : centreY;
  const ratio = bounded / previous;
  mapView.panX = anchorX - centreX - (anchorX - centreX - mapView.panX) * ratio;
  mapView.panY = anchorY - centreY - (anchorY - centreY - mapView.panY) * ratio;
  mapView.zoom = bounded;
  clampMapPan(rect.width, rect.height);
  if (shouldRender) renderTrack(replayProgress);
}

function canvasPoint(event) {
  const rect = trackCanvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function nearestTelemetry(point, threshold = 24) {
  if (!simulationData || !lastGeometry || !simulationData.telemetry.x.length) return -1;
  const telemetry = simulationData.telemetry;
  const step = Math.max(1, Math.floor(telemetry.x.length / 3500));
  let nearest = -1;
  let bestSquared = threshold * threshold;
  for (let index = 0; index < telemetry.x.length; index += step) {
    const projected = lastGeometry.point(telemetry.x[index], telemetry.y[index]);
    const squared = (projected.x - point.x) ** 2 + (projected.y - point.y) ** 2;
    if (squared <= bestSquared) {
      nearest = index;
      bestSquared = squared;
    }
  }
  return nearest;
}

function telemetryFallback(kind, lap) {
  if (!lastPlan) return "";
  const item = kind === "tyre" ? activePhase(lastPlan.stints, lap) : activePhase(lastPlan.weather, lap);
  return kind === "tyre" ? item.tyre : item.condition;
}

function positionTooltip(point) {
  const width = 230;
  const height = 150;
  const x = Math.max(12, Math.min(trackStage.clientWidth - width - 12, point.x + 18));
  const y = Math.max(12, Math.min(trackStage.clientHeight - height - 12, point.y + 18));
  mapTooltip.style.left = `${x}px`;
  mapTooltip.style.top = `${y}px`;
}

function showTelemetry(index, pointerPoint, pinned = false) {
  if (!simulationData || index < 0) {
    if (!tooltipPinned) mapTooltip.hidden = true;
    return;
  }
  const telemetry = simulationData.telemetry;
  const bounded = Math.min(index, telemetry.x.length - 1);
  const lap = Number.isFinite(telemetry.lap[bounded]) ? Math.max(1, Math.round(telemetry.lap[bounded])) : 1;
  const time = Number.isFinite(telemetry.time[bounded]) ? telemetry.time[bounded] : 0;
  const speed = telemetry.speed[bounded];
  const clearance = telemetry.clearance[bounded];
  const tyreToken = telemetry.tyre[bounded] || telemetryFallback("tyre", lap);
  const weatherToken = telemetry.weather[bounded] || telemetryFallback("weather", lap);
  document.getElementById("tooltipLap").textContent = `Lap ${lap}`;
  document.getElementById("tooltipTime").textContent = formatDuration(time, true);
  document.getElementById("tooltipSpeed").textContent = Number.isFinite(speed) ? `${speed.toFixed(1)} km/h` : "Not supplied";
  document.getElementById("tooltipClearance").textContent = Number.isFinite(clearance) ? `${clearance.toFixed(2)} m` : "Not supplied";
  document.getElementById("tooltipTyre").textContent = TYRES[tyreToken] ? TYRES[tyreToken].label : readableToken(tyreToken) || "Not supplied";
  document.getElementById("tooltipWeather").textContent = WEATHER[weatherToken] ? WEATHER[weatherToken].label : readableToken(weatherToken) || "Not supplied";
  document.getElementById("tooltipHint").textContent = pinned ? "Click map to unpin" : "Click to pin";
  hoveredTelemetryIndex = bounded;
  mapTooltip.hidden = false;
  positionTooltip(pointerPoint);
  renderTrack(replayProgress);
}

function updateHover(point) {
  if (tooltipPinned || dragMoved) return;
  const index = nearestTelemetry(point);
  if (index >= 0) showTelemetry(index, point, false);
  else {
    hoveredTelemetryIndex = -1;
    mapTooltip.hidden = true;
    renderTrack(replayProgress);
  }
}

function pointerPairState() {
  const points = [...activePointers.values()];
  if (points.length < 2) return null;
  const first = points[0];
  const second = points[1];
  return {
    distance: Math.hypot(second.x - first.x, second.y - first.y),
    centre: { x: (first.x + second.x) / 2, y: (first.y + second.y) / 2 }
  };
}

trackCanvas.addEventListener("wheel", event => {
  event.preventDefault();
  const point = canvasPoint(event);
  const factor = Math.exp(-event.deltaY * 0.0015);
  zoomAt(mapView.zoom * factor, point);
}, { passive: false });

trackCanvas.addEventListener("dblclick", event => {
  event.preventDefault();
  zoomAt(mapView.zoom * 1.8, canvasPoint(event));
});

trackCanvas.addEventListener("pointerdown", event => {
  const isNewGesture = activePointers.size === 0;
  const point = canvasPoint(event);
  activePointers.set(event.pointerId, point);
  trackCanvas.setPointerCapture(event.pointerId);
  if (isNewGesture) dragMoved = false;
  if (activePointers.size === 1) dragOrigin = point;
  if (activePointers.size >= 2) pinchState = pointerPairState();
});

trackCanvas.addEventListener("pointermove", event => {
  const point = canvasPoint(event);
  if (!activePointers.has(event.pointerId)) {
    updateHover(point);
    return;
  }
  activePointers.set(event.pointerId, point);
  if (activePointers.size >= 2) {
    const current = pointerPairState();
    if (current && pinchState && pinchState.distance > 0) {
      zoomAt(mapView.zoom * (current.distance / pinchState.distance), pinchState.centre, false);
      mapView.panX += current.centre.x - pinchState.centre.x;
      mapView.panY += current.centre.y - pinchState.centre.y;
      const rect = trackCanvas.getBoundingClientRect();
      clampMapPan(rect.width, rect.height);
      pinchState = current;
      dragMoved = true;
      trackCanvas.classList.add("is-dragging");
      renderTrack(replayProgress);
    }
    return;
  }
  if (dragOrigin) {
    const dx = point.x - dragOrigin.x;
    const dy = point.y - dragOrigin.y;
    if (Math.hypot(dx, dy) > 2) dragMoved = true;
    if (dragMoved) {
      mapView.panX += dx;
      mapView.panY += dy;
      dragOrigin = point;
      const rect = trackCanvas.getBoundingClientRect();
      clampMapPan(rect.width, rect.height);
      trackCanvas.classList.add("is-dragging");
      if (!tooltipPinned) mapTooltip.hidden = true;
      renderTrack(replayProgress);
    }
  }
});

function finishPointer(event) {
  const point = canvasPoint(event);
  const wasClick = activePointers.size === 1 && !dragMoved;
  activePointers.delete(event.pointerId);
  if (trackCanvas.hasPointerCapture(event.pointerId)) trackCanvas.releasePointerCapture(event.pointerId);
  trackCanvas.classList.remove("is-dragging");
  dragOrigin = activePointers.size === 1 ? [...activePointers.values()][0] : null;
  pinchState = activePointers.size >= 2 ? pointerPairState() : null;
  if (wasClick) {
    const index = nearestTelemetry(point);
    if (tooltipPinned) {
      tooltipPinned = false;
      hoveredTelemetryIndex = -1;
      mapTooltip.hidden = true;
      renderTrack(replayProgress);
    } else if (index >= 0) {
      tooltipPinned = true;
      showTelemetry(index, point, true);
    }
  }
  if (activePointers.size === 0) dragMoved = false;
}

trackCanvas.addEventListener("pointerup", finishPointer);
trackCanvas.addEventListener("pointercancel", finishPointer);
trackCanvas.addEventListener("pointerleave", () => {
  if (!tooltipPinned && activePointers.size === 0) {
    hoveredTelemetryIndex = -1;
    mapTooltip.hidden = true;
    renderTrack(replayProgress);
  }
});

trackCanvas.addEventListener("keydown", event => {
  const rect = trackCanvas.getBoundingClientRect();
  const panStep = 42;
  let handled = true;
  if (event.key === "+" || event.key === "=") zoomAt(mapView.zoom * 1.35);
  else if (event.key === "-" || event.key === "_") zoomAt(mapView.zoom / 1.35);
  else if (event.key === "0" || event.key === "Home") resetMapView();
  else if (event.key === "ArrowLeft") mapView.panX += panStep;
  else if (event.key === "ArrowRight") mapView.panX -= panStep;
  else if (event.key === "ArrowUp") mapView.panY += panStep;
  else if (event.key === "ArrowDown") mapView.panY -= panStep;
  else handled = false;
  if (handled) {
    event.preventDefault();
    clampMapPan(rect.width, rect.height);
    renderTrack(replayProgress);
  }
});

function drawLapChart(lapTimes) {
  const dimensions = fitCanvas(chartCanvas, chartContext);
  const width = dimensions.width;
  const height = dimensions.height;
  chartContext.clearRect(0, 0, width, height);
  if (!lapTimes.length) {
    chartContext.fillStyle = "#9ea9bc";
    chartContext.font = "13px system-ui, sans-serif";
    chartContext.textAlign = "center";
    chartContext.fillText("No completed lap times", width / 2, height / 2);
    return;
  }

  const padding = { top: 24, right: 18, bottom: 38, left: 58 };
  const graphWidth = Math.max(1, width - padding.left - padding.right);
  const graphHeight = Math.max(1, height - padding.top - padding.bottom);
  const minimumValue = Math.min(...lapTimes);
  const maximumValue = Math.max(...lapTimes);
  const spread = Math.max(0.5, maximumValue - minimumValue);
  const axisMin = Math.max(0, minimumValue - spread * 0.35);
  const axisMax = maximumValue + spread * 0.25;

  chartContext.save();
  chartContext.font = "11px system-ui, sans-serif";
  chartContext.textBaseline = "middle";
  for (let tick = 0; tick <= 4; tick += 1) {
    const ratio = tick / 4;
    const y = padding.top + graphHeight * ratio;
    const value = axisMax - (axisMax - axisMin) * ratio;
    chartContext.strokeStyle = "rgba(255, 255, 255, 0.07)";
    chartContext.lineWidth = 1;
    chartContext.beginPath();
    chartContext.moveTo(padding.left, Math.round(y) + 0.5);
    chartContext.lineTo(width - padding.right, Math.round(y) + 0.5);
    chartContext.stroke();
    chartContext.fillStyle = "#7f899b";
    chartContext.textAlign = "right";
    chartContext.fillText(`${value.toFixed(1)}s`, padding.left - 9, y);
  }

  const slotWidth = graphWidth / lapTimes.length;
  const barWidth = Math.max(5, Math.min(42, slotWidth * 0.62));
  lapTimes.forEach((time, index) => {
    const ratio = (time - axisMin) / Math.max(0.001, axisMax - axisMin);
    const barHeight = Math.max(3, ratio * graphHeight);
    const x = padding.left + slotWidth * index + (slotWidth - barWidth) / 2;
    const y = padding.top + graphHeight - barHeight;
    const isFastest = Math.abs(time - minimumValue) < 0.0005;
    const gradient = chartContext.createLinearGradient(0, y, 0, padding.top + graphHeight);
    gradient.addColorStop(0, isFastest ? "#61d095" : "#ff4057");
    gradient.addColorStop(1, isFastest ? "rgba(97, 208, 149, 0.28)" : "rgba(216, 23, 53, 0.25)");
    chartContext.fillStyle = gradient;
    roundedRect(chartContext, x, y, barWidth, barHeight, Math.min(6, barWidth / 3));
    chartContext.fill();

    chartContext.fillStyle = isFastest ? "#aaf4c9" : "#9ea9bc";
    chartContext.textAlign = "center";
    chartContext.textBaseline = "top";
    const labelEvery = lapTimes.length <= 12 ? 1 : Math.ceil(lapTimes.length / 10);
    if (index % labelEvery === 0 || index === lapTimes.length - 1) chartContext.fillText(String(index + 1), x + barWidth / 2, height - padding.bottom + 12);
  });
  chartContext.restore();
  chartCanvas.setAttribute("aria-label", `Lap-time chart for ${lapTimes.length} completed ${lapTimes.length === 1 ? "lap" : "laps"}. Fastest lap ${formatDuration(minimumValue)}.`);
}

function tracePoints(data) {
  const baselineTarget = [];
  const targetCount = Math.min(data.trajectory.speed.length, data.trajectory.x.length);
  const suppliedDistances = data.trajectory.distance.filter(Number.isFinite);
  const maximumSuppliedDistance = suppliedDistances.length ? Math.max(...suppliedDistances) : NaN;
  const lapDistance = Number.isFinite(data.track.lengthM) && data.track.lengthM > 0
    ? data.track.lengthM
    : maximumSuppliedDistance;
  const hasDistance = Number.isFinite(lapDistance) && lapDistance > 0
    && targetCount >= 2
    && data.trajectory.distance.slice(0, targetCount).every(Number.isFinite);
  for (let index = 0; index < targetCount; index += 1) {
    const speed = data.trajectory.speed[index];
    if (!Number.isFinite(speed)) continue;
    const supplied = data.trajectory.distance[index];
    const progress = hasDistance && Number.isFinite(supplied)
      ? supplied / lapDistance
      : targetCount > 1 ? index / (targetCount - 1) : 0;
    if (Number.isFinite(progress)) baselineTarget.push({ progress: Math.max(0, Math.min(1, progress)), speed });
  }

  const lapTimes = data.summary.lapTimes;
  const fastestLapIndex = lapTimes.length
    ? lapTimes.indexOf(Math.min(...lapTimes))
    : -1;
  const validProgress = data.telemetry.progress.filter(Number.isFinite);
  const maximumProgress = validProgress.length ? Math.max(...validProgress) : 0;
  const traceLapIndex = fastestLapIndex >= 0
    ? fastestLapIndex
    : Math.max(0, Math.floor(Math.max(0, maximumProgress - 1.0e-6)));
  let target = [];
  let simulated = [];
  for (let index = 0; index < data.telemetry.speed.length; index += 1) {
    const speed = data.telemetry.speed[index];
    const targetSpeed = data.telemetry.targetSpeed[index];
    const progressLaps = data.telemetry.progress[index];
    if (!Number.isFinite(speed) || !Number.isFinite(progressLaps)) continue;
    const progress = progressLaps - traceLapIndex;
    if (progress < -1.0e-6 || progress > 1 + 1.0e-6) continue;
    const clampedProgress = Math.max(0, Math.min(1, progress));
    simulated.push({ progress: clampedProgress, speed });
    if (Number.isFinite(targetSpeed)) {
      target.push({ progress: clampedProgress, speed: targetSpeed });
    }
  }

  if (simulated.length < 2) {
    const lapNumber = traceLapIndex + 1;
    const speeds = data.telemetry.speed.filter((speed, index) => Number.isFinite(speed)
      && Math.round(data.telemetry.lap[index]) === lapNumber);
    if (speeds.length >= 2) {
      simulated = speeds.map((speed, index) => ({
        progress: index / (speeds.length - 1),
        speed
      }));
    }
  }

  if (simulated.length < 2) {
    const speeds = data.telemetry.speed.filter(Number.isFinite);
    if (speeds.length >= 2) {
      simulated = speeds.map((speed, index) => ({
        progress: index / (speeds.length - 1),
        speed
      }));
    }
  }

  if (target.length < 2) target = baselineTarget;

  target.sort((a, b) => a.progress - b.progress);
  simulated.sort((a, b) => a.progress - b.progress);
  return {
    target,
    simulated,
    lapNumber: traceLapIndex + 1,
    lapDistance: Number.isFinite(lapDistance) && lapDistance > 0 ? lapDistance : NaN
  };
}

function drawSpeedSeries(context, points, xAt, yAt, colour, width, dash = []) {
  if (!points.length) return;
  context.save();
  context.strokeStyle = colour;
  context.lineWidth = width;
  context.lineJoin = "round";
  context.lineCap = "round";
  context.setLineDash(dash);
  context.beginPath();
  points.forEach((point, index) => {
    const x = xAt(point.progress);
    const y = yAt(point.speed);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();
  context.restore();
}

function drawSpeedTrace() {
  if (!speedTraceCanvas || !speedTraceContext) return;
  const dimensions = fitCanvas(speedTraceCanvas, speedTraceContext);
  const width = dimensions.width;
  const height = dimensions.height;
  speedTraceContext.clearRect(0, 0, width, height);
  if (!simulationData) {
    speedTraceContext.fillStyle = "#9ea9bc";
    speedTraceContext.font = "13px system-ui, sans-serif";
    speedTraceContext.textAlign = "center";
    speedTraceContext.fillText("Run a simulation to compare speed traces", width / 2, height / 2);
    return;
  }

  const series = tracePoints(simulationData);
  const speeds = [...series.target, ...series.simulated].map(point => point.speed).filter(Number.isFinite);
  if (!speeds.length) {
    speedTraceContext.fillStyle = "#9ea9bc";
    speedTraceContext.font = "13px system-ui, sans-serif";
    speedTraceContext.textAlign = "center";
    speedTraceContext.fillText("No speed samples were returned", width / 2, height / 2);
    speedTraceCanvas.setAttribute("aria-label", "Speed trace unavailable because no speed samples were returned.");
    return;
  }

  const compact = width < 520;
  const padding = { top: 24, right: compact ? 10 : 20, bottom: 48, left: compact ? 47 : 60 };
  const graphWidth = Math.max(1, width - padding.left - padding.right);
  const graphHeight = Math.max(1, height - padding.top - padding.bottom);
  const minimumSpeed = Math.min(...speeds);
  const maximumSpeed = Math.max(...speeds);
  const axisMin = Math.max(0, Math.floor((minimumSpeed - 15) / 20) * 20);
  const axisMax = Math.max(axisMin + 20, Math.ceil((maximumSpeed + 10) / 20) * 20);
  const xAt = progress => padding.left + Math.max(0, Math.min(1, progress)) * graphWidth;
  const yAt = speed => padding.top + (axisMax - speed) / (axisMax - axisMin) * graphHeight;

  speedTraceContext.save();
  speedTraceContext.font = `${compact ? 10 : 11}px system-ui, sans-serif`;
  speedTraceContext.textBaseline = "middle";
  for (let tick = 0; tick <= 4; tick += 1) {
    const ratio = tick / 4;
    const y = padding.top + graphHeight * ratio;
    const value = axisMax - (axisMax - axisMin) * ratio;
    speedTraceContext.strokeStyle = "rgba(255, 255, 255, 0.07)";
    speedTraceContext.lineWidth = 1;
    speedTraceContext.beginPath();
    speedTraceContext.moveTo(padding.left, Math.round(y) + 0.5);
    speedTraceContext.lineTo(width - padding.right, Math.round(y) + 0.5);
    speedTraceContext.stroke();
    speedTraceContext.fillStyle = "#7f899b";
    speedTraceContext.textAlign = "right";
    speedTraceContext.fillText(`${Math.round(value)}`, padding.left - 8, y);
  }

  for (let tick = 0; tick <= 4; tick += 1) {
    const ratio = tick / 4;
    const x = xAt(ratio);
    speedTraceContext.strokeStyle = "rgba(255, 255, 255, 0.045)";
    speedTraceContext.beginPath();
    speedTraceContext.moveTo(Math.round(x) + 0.5, padding.top);
    speedTraceContext.lineTo(Math.round(x) + 0.5, padding.top + graphHeight);
    speedTraceContext.stroke();
    speedTraceContext.fillStyle = "#7f899b";
    speedTraceContext.textAlign = tick === 0 ? "left" : tick === 4 ? "right" : "center";
    const label = Number.isFinite(series.lapDistance)
      ? `${(series.lapDistance * ratio / 1000).toFixed(series.lapDistance < 4000 ? 2 : 1)}`
      : `${Math.round(ratio * 100)}%`;
    speedTraceContext.fillText(label, x, height - padding.bottom + 17);
  }

  speedTraceContext.fillStyle = "#9ea9bc";
  speedTraceContext.textAlign = "center";
  speedTraceContext.fillText(Number.isFinite(series.lapDistance) ? "Lap distance (km)" : "Lap progress", padding.left + graphWidth / 2, height - 8);
  speedTraceContext.save();
  speedTraceContext.translate(12, padding.top + graphHeight / 2);
  speedTraceContext.rotate(-Math.PI / 2);
  speedTraceContext.fillText("Speed (km/h)", 0, 0);
  speedTraceContext.restore();

  speedTraceContext.beginPath();
  speedTraceContext.rect(padding.left, padding.top, graphWidth, graphHeight);
  speedTraceContext.clip();
  drawSpeedSeries(speedTraceContext, series.target, xAt, yAt, "#ffd166", 2, [7, 5]);
  drawSpeedSeries(speedTraceContext, series.simulated, xAt, yAt, "rgba(51, 214, 208, 0.2)", 6);
  drawSpeedSeries(speedTraceContext, series.simulated, xAt, yAt, "#33d6d0", 2.2);
  speedTraceContext.restore();

  const simulatedDescription = series.simulated.length
    ? `simulated lap ${series.lapNumber}`
    : "no simulated samples";
  const targetDescription = series.target.length ? "optimized target" : "no target samples";
  speedTraceCanvas.setAttribute("aria-label", `Speed trace comparing ${simulatedDescription} with the ${targetDescription} around ${simulationData.track.name}.`);
}

function roundedRect(context, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + r, y);
  context.arcTo(x + width, y, x + width, y + height, r);
  context.arcTo(x + width, y + height, x, y + height, r);
  context.arcTo(x, y + height, x, y, r);
  context.arcTo(x, y, x + width, y, r);
  context.closePath();
}

function percentageValue(source, percentKeys, ratioKeys = []) {
  for (const key of percentKeys) {
    const value = Number(source?.[key]);
    if (Number.isFinite(value)) return value;
  }
  for (const key of ratioKeys) {
    const value = Number(source?.[key]);
    if (Number.isFinite(value)) return value * 100;
  }
  return NaN;
}

function normalizeTyreLap(raw, index) {
  if (!raw || typeof raw !== "object") return null;
  const lap = Math.max(1, Math.round(finiteNumber(firstDefined(raw.lap, raw.lap_number, raw.lapNumber), index + 1)));
  const point = {
    lap,
    compound: String(firstDefined(raw.compound, raw.tyre, raw.tyre_compound, "")),
    wear: percentageValue(raw, ["wear_index_pct", "wear_pct", "wear_percent"], ["wear_index", "wear_ratio"]),
    performance: percentageValue(
      raw,
      ["performance_remaining_pct", "performance_pct", "remaining_performance_pct"],
      ["performance_remaining", "performance_ratio"]
    ),
    surfaceTemp: finiteNumber(firstDefined(
      raw.estimated_surface_temp_c,
      raw.surface_temp_c,
      raw.tyre_surface_temp_c,
      raw.surface_temperature_c
    ), NaN),
    coreTemp: finiteNumber(firstDefined(
      raw.estimated_core_temp_c,
      raw.core_temp_c,
      raw.tyre_core_temp_c,
      raw.core_temperature_c
    ), NaN),
    paceLoss: finiteNumber(firstDefined(raw.pace_loss_s, raw.estimated_pace_loss_s, raw.lap_time_loss_s), NaN)
  };
  return Object.values(point).some((value, valueIndex) => valueIndex > 1 && Number.isFinite(value)) ? point : null;
}

function normalizeTyreAnalysis(raw) {
  if (!raw || typeof raw !== "object") return null;
  const rawLaps = firstDefined(raw.laps, raw.per_lap, raw.lap_analysis, raw.lap_points, []);
  const laps = Array.isArray(rawLaps) ? rawLaps.map(normalizeTyreLap).filter(Boolean).sort((a, b) => a.lap - b.lap) : [];
  const stints = Array.isArray(firstDefined(raw.stints, raw.stint_analysis))
    ? firstDefined(raw.stints, raw.stint_analysis)
    : [];
  const causes = Array.isArray(firstDefined(raw.causes, raw.wear_causes, raw.contributors, raw.explanations))
    ? firstDefined(raw.causes, raw.wear_causes, raw.contributors, raw.explanations)
    : [];
  const recommendations = Array.isArray(firstDefined(raw.recommendations, raw.strategy_recommendations, raw.suggestions))
    ? firstDefined(raw.recommendations, raw.strategy_recommendations, raw.suggestions)
    : [];
  const overview = firstDefined(raw.overview, raw.summary, {});
  return {
    raw,
    laps,
    stints,
    causes,
    recommendations,
    overview: overview && typeof overview === "object" ? overview : {},
    message: typeof firstDefined(raw.review, raw.message, typeof overview === "string" ? overview : null) === "string"
      ? firstDefined(raw.review, raw.message, overview)
      : "",
    cliffLap: finiteNumber(firstDefined(
      raw.projected_cliff_lap,
      raw.cliff_lap,
      overview?.projected_cliff_lap,
      overview?.cliff_lap
    ), NaN)
  };
}

function tyreMetric(values, mode = "maximum") {
  const valid = values.filter(Number.isFinite);
  if (!valid.length) return NaN;
  if (mode === "last") return valid[valid.length - 1];
  return Math.max(...valid);
}

function insightText(item, recommendation = false) {
  if (typeof item === "string") return { title: item, detail: "", meta: "" };
  if (!item || typeof item !== "object") return null;
  const title = String(firstDefined(
    item.title,
    recommendation ? item.action : item.cause,
    recommendation ? item.recommendation : item.summary,
    item.message,
    item.label,
    ""
  )).trim();
  const detail = String(firstDefined(
    recommendation ? item.rationale : item.explanation,
    recommendation ? item.reason : item.detail,
    item.description,
    item.why,
    ""
  )).trim();
  const meta = [];
  const gain = finiteNumber(firstDefined(item.estimated_gain_s, item.projected_gain_s, item.time_gain_s), NaN);
  if (Number.isFinite(gain)) meta.push(`${gain >= 0 ? "+" : "\u2212"}${Math.abs(gain).toFixed(2)} s projected`);
  const contribution = percentageValue(item, ["contribution_pct", "impact_pct"], ["contribution", "impact"]);
  if (Number.isFinite(contribution)) meta.push(`${contribution.toFixed(0)}% contribution`);
  const confidence = firstDefined(item.confidence, item.confidence_label);
  if (typeof confidence === "string" && confidence.trim()) meta.push(`${readableToken(confidence)} confidence`);
  if (!title && !detail) return null;
  return { title: title || detail, detail: title ? detail : "", meta: meta.join(" \u00b7 ") };
}

function renderInsightList(container, items, recommendation = false) {
  container.replaceChildren();
  for (const item of items) {
    const insight = insightText(item, recommendation);
    if (!insight) continue;
    const listItem = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = insight.title;
    listItem.append(title);
    if (insight.detail) {
      const detail = document.createElement("span");
      detail.textContent = insight.detail;
      listItem.append(detail);
    }
    if (insight.meta) {
      const meta = document.createElement("small");
      meta.textContent = insight.meta;
      listItem.append(meta);
    }
    container.append(listItem);
  }
}

function stintValue(stint, percentKeys, ratioKeys = []) {
  return percentageValue(stint, percentKeys, ratioKeys);
}

function renderTyreStints(stints) {
  tyreStintCards.replaceChildren();
  for (const [index, stint] of stints.entries()) {
    if (!stint || typeof stint !== "object") continue;
    const compound = String(firstDefined(stint.compound, stint.tyre, stint.tyre_compound, "Tyre"));
    const startLap = Math.max(1, Math.round(finiteNumber(firstDefined(stint.start_lap, stint.startLap), index + 1)));
    const endLap = Math.max(startLap, Math.round(finiteNumber(firstDefined(stint.end_lap, stint.endLap), startLap)));
    const wear = stintValue(stint, ["end_wear_index_pct", "wear_index_pct", "wear_pct"], ["end_wear_index", "wear_index"]);
    const performance = stintValue(
      stint,
      ["end_performance_remaining_pct", "performance_remaining_pct", "performance_pct"],
      ["end_performance_remaining", "performance_remaining"]
    );
    const paceLoss = finiteNumber(firstDefined(stint.pace_loss_s, stint.end_pace_loss_s, stint.estimated_pace_loss_s), NaN);
    const peakTemp = finiteNumber(firstDefined(stint.peak_surface_temp_c, stint.max_surface_temp_c, stint.estimated_surface_temp_c), NaN);
    const coreTemp = finiteNumber(firstDefined(stint.average_core_temp_c, stint.estimated_core_temp_c, stint.core_temp_c), NaN);
    const safeCompound = compound.toLowerCase().replace(/[^a-z0-9_-]+/g, "-");
    const card = document.createElement("article");
    card.className = `tyre-stint-card tyre-${safeCompound}`;
    const heading = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = readableToken(compound);
    const range = document.createElement("span");
    range.textContent = startLap === endLap ? `Lap ${startLap}` : `Laps ${startLap}\u2013${endLap}`;
    heading.append(title, range);
    const metrics = document.createElement("dl");
    const values = [
      ["End wear", Number.isFinite(wear) ? `${wear.toFixed(1)}%` : "\u2014"],
      ["Performance", Number.isFinite(performance) ? `${performance.toFixed(1)}%` : "\u2014"],
      ["Peak surface", Number.isFinite(peakTemp) ? `${peakTemp.toFixed(1)} \u00b0C` : "\u2014"],
      ["Average core", Number.isFinite(coreTemp) ? `${coreTemp.toFixed(1)} \u00b0C` : "\u2014"],
      ["Pace loss", Number.isFinite(paceLoss) ? `${paceLoss.toFixed(3)} s` : "\u2014"]
    ];
    for (const [label, value] of values) {
      const item = document.createElement("div");
      const term = document.createElement("dt");
      const description = document.createElement("dd");
      term.textContent = label;
      description.textContent = value;
      item.append(term, description);
      metrics.append(item);
    }
    card.append(heading, metrics);
    tyreStintCards.append(card);
  }
}

function drawTyreWearChart(points) {
  if (!tyreWearCanvas || !tyreWearContext) return;
  const dimensions = fitCanvas(tyreWearCanvas, tyreWearContext);
  const { width, height } = dimensions;
  tyreWearContext.clearRect(0, 0, width, height);
  if (!points.length) return;

  const compact = width < 600;
  const padding = { top: 45, right: compact ? 45 : 58, bottom: 42, left: compact ? 42 : 52 };
  const graphWidth = Math.max(1, width - padding.left - padding.right);
  const graphHeight = Math.max(1, height - padding.top - padding.bottom);
  const lapMin = Math.min(...points.map(point => point.lap));
  const lapMax = Math.max(...points.map(point => point.lap));
  const temperatures = points.flatMap(point => [point.surfaceTemp, point.coreTemp]).filter(Number.isFinite);
  const temperatureMin = temperatures.length ? Math.floor((Math.min(...temperatures) - 5) / 10) * 10 : 50;
  const temperatureMax = temperatures.length ? Math.max(temperatureMin + 20, Math.ceil((Math.max(...temperatures) + 5) / 10) * 10) : 130;
  const xAt = lap => padding.left + (lapMax === lapMin ? 0.5 : (lap - lapMin) / (lapMax - lapMin)) * graphWidth;
  const percentY = value => padding.top + (100 - Math.max(0, Math.min(100, value))) / 100 * graphHeight;
  const temperatureY = value => padding.top + (temperatureMax - value) / (temperatureMax - temperatureMin) * graphHeight;

  tyreWearContext.save();
  tyreWearContext.font = `${compact ? 9 : 10}px system-ui, sans-serif`;
  tyreWearContext.textBaseline = "middle";
  for (let tick = 0; tick <= 4; tick += 1) {
    const percent = 100 - tick * 25;
    const y = padding.top + graphHeight * tick / 4;
    tyreWearContext.strokeStyle = "rgba(255, 255, 255, 0.07)";
    tyreWearContext.beginPath();
    tyreWearContext.moveTo(padding.left, Math.round(y) + 0.5);
    tyreWearContext.lineTo(width - padding.right, Math.round(y) + 0.5);
    tyreWearContext.stroke();
    tyreWearContext.fillStyle = "#7f899b";
    tyreWearContext.textAlign = "right";
    tyreWearContext.fillText(`${percent}%`, padding.left - 7, y);
    const temperature = temperatureMax - (temperatureMax - temperatureMin) * tick / 4;
    tyreWearContext.textAlign = "left";
    tyreWearContext.fillText(`${temperature.toFixed(0)}\u00b0`, width - padding.right + 7, y);
  }
  const labelLaps = [...new Set(points.map(point => point.lap))];
  const labelStride = Math.max(1, Math.ceil(labelLaps.length / (compact ? 5 : 8)));
  labelLaps.forEach((lap, index) => {
    if (index % labelStride !== 0 && index !== labelLaps.length - 1) return;
    tyreWearContext.fillStyle = "#7f899b";
    tyreWearContext.textAlign = "center";
    tyreWearContext.fillText(`L${lap}`, xAt(lap), height - padding.bottom + 17);
  });

  const lines = [
    { key: "performance", colour: "#33d6d0", y: percentY, label: "Performance" },
    { key: "wear", colour: "#ff4057", y: percentY, label: "Wear" },
    { key: "surfaceTemp", colour: "#ffb04a", y: temperatureY, label: "Surface temp." },
    { key: "coreTemp", colour: "#a78bfa", y: temperatureY, label: "Core temp." }
  ];
  lines.forEach((line, lineIndex) => {
    const values = points.filter(point => Number.isFinite(point[line.key]));
    if (!values.length) return;
    tyreWearContext.strokeStyle = line.colour;
    tyreWearContext.lineWidth = 2.2;
    tyreWearContext.lineJoin = "round";
    tyreWearContext.beginPath();
    values.forEach((point, index) => {
      const x = xAt(point.lap);
      const y = line.y(point[line.key]);
      if (index === 0) tyreWearContext.moveTo(x, y);
      else tyreWearContext.lineTo(x, y);
    });
    tyreWearContext.stroke();
    const legendColumns = compact ? 2 : 4;
    const legendWidth = graphWidth / legendColumns;
    const legendX = padding.left + (lineIndex % legendColumns) * legendWidth;
    const legendY = 13 + Math.floor(lineIndex / legendColumns) * 17;
    tyreWearContext.fillStyle = line.colour;
    tyreWearContext.fillRect(legendX, legendY + 3, 18, 3);
    tyreWearContext.fillStyle = "#aeb7c7";
    tyreWearContext.textAlign = "left";
    tyreWearContext.fillText(line.label, legendX + 24, legendY + 5);
  });
  tyreWearContext.restore();
  tyreWearCanvas.setAttribute("aria-label", `Estimated tyre wear, remaining performance, surface temperature, and core temperature across laps ${lapMin} to ${lapMax}.`);
}

function populateTyreReview(rawAnalysis, unavailableMessage = "Tyre analysis is unavailable for this run.") {
  const analysis = normalizeTyreAnalysis(rawAnalysis);
  tyreLapPoints = analysis?.laps || [];
  tyreStintCards.replaceChildren();
  tyreCauseList.replaceChildren();
  tyreRecommendationList.replaceChildren();
  tyreOverview.hidden = true;
  tyreChartWrap.hidden = true;
  tyreReviewDetails.hidden = true;

  if (!analysis || (!analysis.laps.length && !analysis.stints.length && !analysis.causes.length && !analysis.recommendations.length)) {
    tyreReviewMessage.hidden = false;
    tyreReviewMessage.textContent = unavailableMessage;
    if (tyreWearCanvas && tyreWearContext) {
      const dimensions = fitCanvas(tyreWearCanvas, tyreWearContext);
      tyreWearContext.clearRect(0, 0, dimensions.width, dimensions.height);
    }
    return;
  }

  const overview = analysis.overview;
  const peakWear = firstFinite(
    percentageValue(overview, ["peak_wear_index_pct", "peak_wear_pct"], ["peak_wear_index"]),
    tyreMetric(analysis.laps.map(point => point.wear))
  );
  const endPerformance = firstFinite(
    percentageValue(overview, ["end_performance_remaining_pct", "performance_remaining_pct"], ["performance_remaining"]),
    tyreMetric(analysis.laps.map(point => point.performance), "last")
  );
  const peakSurfaceTemp = firstFinite(
    overview.peak_surface_temp_c,
    overview.max_surface_temp_c,
    tyreMetric(analysis.laps.map(point => point.surfaceTemp))
  );
  const paceLoss = firstFinite(
    overview.estimated_pace_loss_s,
    overview.pace_loss_s,
    tyreMetric(analysis.laps.map(point => point.paceLoss))
  );
  const cliffLap = analysis.cliffLap;
  document.getElementById("tyreWearPeak").textContent = Number.isFinite(peakWear) ? `${peakWear.toFixed(1)}%` : "\u2014";
  document.getElementById("tyrePerformanceEnd").textContent = Number.isFinite(endPerformance) ? `${endPerformance.toFixed(1)}%` : "\u2014";
  document.getElementById("tyreTemperaturePeak").textContent = Number.isFinite(peakSurfaceTemp) ? `${peakSurfaceTemp.toFixed(1)} \u00b0C` : "\u2014";
  document.getElementById("tyrePaceLoss").textContent = Number.isFinite(paceLoss) ? `${paceLoss.toFixed(3)} s/lap` : "\u2014";
  document.getElementById("tyreCliffLap").textContent = Number.isFinite(cliffLap) ? `Lap ${Math.round(cliffLap)}` : "Not reached";
  tyreOverview.hidden = false;

  tyreReviewMessage.hidden = false;
  tyreReviewMessage.textContent = analysis.message || "Estimated tyre condition is shown by lap and stint, with the largest modelled wear contributors ranked below.";
  if (analysis.laps.length) {
    tyreChartWrap.hidden = false;
    requestAnimationFrame(() => drawTyreWearChart(tyreLapPoints));
  }
  renderTyreStints(analysis.stints);
  renderInsightList(tyreCauseList, analysis.causes);
  renderInsightList(tyreRecommendationList, analysis.recommendations, true);
  const detailSections = [tyreStintCards, tyreCauseList, tyreRecommendationList];
  detailSections.forEach(container => {
    const section = container.closest("section");
    if (section) section.hidden = container.children.length === 0;
  });
  tyreReviewDetails.hidden = detailSections.every(container => container.children.length === 0);
}

function setSavedRunContext(entry = null, summaryOnly = false) {
  viewingSavedEntry = entry;
  if (!entry) {
    savedRunContext.hidden = true;
    savedRunContextText.textContent = "";
    resultsEyebrow.textContent = "Latest simulation";
    return;
  }
  resultsEyebrow.textContent = "Saved simulation";
  savedRunContext.hidden = false;
  savedRunContextText.textContent = summaryOnly
    ? `Viewing the saved summary from ${historyDate(entry.createdAt)}. Detailed replay and tyre telemetry were not stored for this earlier run.`
    : `Viewing saved results from ${historyDate(entry.createdAt)}.`;
  rerunSavedButton.hidden = !lastPlan;
}

function safeExecutedPlan(plan, laps) {
  const source = plan && typeof plan === "object" ? plan : {};
  const stints = Array.isArray(source.stints) && source.stints.length
    ? source.stints
    : [{ start_lap: 1, tyre: "medium" }];
  const weather = Array.isArray(source.weather) && source.weather.length
    ? source.weather
    : [{ start_lap: 1, condition: "dry", track_temp_c: 25, rain_intensity: 0 }];
  return {
    ...source,
    laps: Math.max(1, Math.round(finiteNumber(source.laps, laps || 1))),
    stints,
    weather
  };
}

function presentLegacyHistory(payload, fallbackEntry) {
  const record = payload.history && typeof payload.history === "object" ? payload.history : fallbackEntry?.source || {};
  const entry = normalizeHistoryEntry(record) || fallbackEntry;
  if (!entry) throw new Error("This saved run does not contain a readable summary.");
  const rawSummary = payload.summary && typeof payload.summary === "object"
    ? payload.summary
    : record.summary && typeof record.summary === "object" ? record.summary : {};
  const summary = normalizeSummary(rawSummary, payload.strategy || record.strategy || null);
  if (!summary.requestedLaps) summary.requestedLaps = entry.requestedLaps;
  if (!summary.completedLaps) summary.completedLaps = entry.completedLaps;
  const plan = safeExecutedPlan(payload.strategy || record.strategy || entry.strategy, summary.requestedLaps);
  const rawTrack = payload.track && typeof payload.track === "object"
    ? payload.track
    : record.circuit && typeof record.circuit === "object" ? record.circuit : {};
  const track = {
    id: String(firstDefined(rawTrack.id, rawTrack.circuit_id, entry.circuitId, "")),
    name: String(firstDefined(rawTrack.name, entry.circuitName, "F1 Circuit")),
    location: String(firstDefined(rawTrack.location, entry.location, "")),
    countryCode: String(firstDefined(rawTrack.country_code, rawTrack.countryCode, entry.countryCode, "")),
    lengthM: finiteNumber(firstDefined(rawTrack.length_m, rawTrack.lengthM), NaN)
  };

  cancelAnimationFrame(replayFrame);
  replayRunning = false;
  simulationData = null;
  lastPlan = plan;
  chartLapTimes = summary.lapTimes;
  document.getElementById("trackTitle").textContent = track.name;
  resultsCircuitLabel.textContent = `${track.name} \u00b7 ${circuitDetails(track)} \u00b7 ${summary.completedLaps}/${summary.requestedLaps} laps completed`;
  trackCanvas.setAttribute("aria-label", `${track.name} saved simulation. Detailed circuit replay was not retained for this legacy result.`);
  const emptyTitle = emptyState.querySelector("strong");
  const emptyDescription = emptyState.querySelector("span:not(.empty-icon)");
  if (emptyTitle) emptyTitle.textContent = "Summary-only saved run";
  if (emptyDescription) emptyDescription.textContent = "Detailed circuit replay was not stored for this earlier simulation.";
  emptyState.classList.remove("is-hidden");
  mapOverlay.classList.remove("is-visible");
  replayButton.classList.remove("is-visible");
  liveBadge.classList.remove("is-visible");
  populateSummary(summary);
  populateLapTable(chartLapTimes, plan);
  populateExecutedPlan(plan, summary);
  populateConsistency(chartLapTimes);
  populateComparison({ track, summary, history: record });
  populateTyreReview(
    firstDefined(payload.tyre_analysis, rawSummary.tyre_analysis, null),
    "Detailed tyre wear analysis was not stored for this earlier simulation. Run the strategy again to generate it."
  );
  lapResults.classList.add("is-visible");
  setSavedRunContext(entry, true);
  resetMapView();
  drawLapChart(chartLapTimes);
  drawSpeedTrace();
}

function presentSimulation(data, plan, options = {}) {
  simulationData = data;
  const executedPlan = data.summary.strategy
    && Array.isArray(data.summary.strategy.stints)
    && Array.isArray(data.summary.strategy.weather)
    ? data.summary.strategy
    : plan;
  lastPlan = safeExecutedPlan(executedPlan, data.summary.requestedLaps);
  chartLapTimes = data.summary.lapTimes;
  if (data.history && !options.savedEntry) {
    const entry = normalizeHistoryEntry(data.history);
    if (entry) {
      historyEntries = [entry, ...historyEntries.filter(item => item.id !== entry.id)];
      renderHistory();
    }
  }
  document.getElementById("trackTitle").textContent = data.track.name;
  const resultDetails = circuitDetails({
    location: data.track.location,
    countryCode: data.track.countryCode,
    lengthM: data.track.lengthM
  });
  resultsCircuitLabel.textContent = `${data.track.name} \u00b7 ${resultDetails} \u00b7 ${data.summary.completedLaps}/${data.summary.requestedLaps} laps completed`;
  trackCanvas.setAttribute("aria-label", `${data.track.name} circuit map showing the optimized racing line and simulated driven path. Use the arrow keys to pan and plus or minus to zoom.`);
  populateSummary(data.summary);
  populateLapTable(chartLapTimes, lastPlan);
  populateExecutedPlan(lastPlan, data.summary);
  populateConsistency(chartLapTimes);
  populateComparison(data);
  populateTyreReview(data.tyreAnalysis);
  setSavedRunContext(options.savedEntry || null, false);
  emptyState.classList.add("is-hidden");
  mapOverlay.classList.add("is-visible");
  replayButton.classList.add("is-visible");
  liveBadge.classList.add("is-visible");
  lapResults.classList.add("is-visible");
  resetMapView();
  drawLapChart(chartLapTimes);
  drawSpeedTrace();
  startReplay();
}

async function runSimulation(plan) {
  const response = await fetch("/api/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Accept": "application/json" },
    body: JSON.stringify({ circuit_id: circuitSelect.value, ...plan })
  });

  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error(response.ok ? "The server returned invalid JSON." : `Simulation request failed (${response.status}).`);
  }
  if (!response.ok) {
    const detail = payload && typeof payload.error === "string"
      ? payload.error
      : payload && typeof payload.detail === "string"
        ? payload.detail
        : `Simulation request failed (${response.status}).`;
    throw new Error(detail);
  }
  return normalizeResponse(payload);
}

function fullHistoryPayload(payload) {
  const candidates = [payload, payload?.result, payload?.simulation, payload?.payload, payload?.details];
  for (const candidate of candidates) {
    if (!candidate || typeof candidate !== "object") continue;
    if (candidate.track && candidate.trajectory && candidate.telemetry && candidate.summary) {
      return {
        ...candidate,
        tyre_analysis: firstDefined(candidate.tyre_analysis, payload?.tyre_analysis, null),
        history: firstDefined(candidate.history, payload?.history, payload?.record, null)
      };
    }
  }
  return null;
}

async function loadHistoryResult(historyId) {
  if (!historyId || historyDetailLoadingId) return;
  const entry = historyEntries.find(item => item.id === historyId);
  if (!entry) {
    showError("That saved simulation is no longer in the history list.");
    return;
  }
  clearError();
  historyDetailLoadingId = historyId;
  renderHistory();
  try {
    const response = await fetch(`/api/history/${encodeURIComponent(historyId)}`, {
      headers: { "Accept": "application/json" }
    });
    let payload;
    try {
      payload = await response.json();
    } catch (error) {
      throw new Error("The saved simulation returned invalid JSON.");
    }
    if (!response.ok) {
      throw new Error(typeof payload?.error === "string" ? payload.error : "The saved simulation could not be loaded.");
    }
    if (payload?.details_available === false) {
      presentLegacyHistory(payload, entry);
    } else {
      const detailed = fullHistoryPayload(payload);
      if (!detailed) throw new Error("This saved run does not contain detailed simulation results.");
      const data = normalizeResponse(detailed);
      const savedEntry = normalizeHistoryEntry(detailed.history) || entry;
      presentSimulation(data, data.summary.strategy || detailed.strategy || entry.strategy, { savedEntry });
    }
    navigateTo("results");
  } catch (error) {
    showError(error instanceof Error ? error.message : "The saved simulation could not be loaded.");
  } finally {
    historyDetailLoadingId = "";
    renderHistory();
  }
}

function loadSavedStrategyIntoBuilder() {
  if (!lastPlan) return;
  const plan = safeExecutedPlan(lastPlan, lastPlan.laps || defaultLaps);
  const laps = Math.max(1, Math.min(maximumLaps, Math.round(finiteNumber(plan.laps, defaultLaps))));
  lapsInput.value = String(laps);
  document.getElementById("requestedLaps").textContent = `of ${laps} ${laps === 1 ? "lap" : "laps"}`;
  if (viewingSavedEntry?.circuitId && circuitCatalog.some(circuit => circuit.id === viewingSavedEntry.circuitId)) {
    circuitSelect.value = viewingSavedEntry.circuitId;
    updateCircuitContext();
  }

  const stints = [...plan.stints].sort((a, b) => finiteNumber(a.start_lap) - finiteNumber(b.start_lap));
  const startingTyre = String(firstDefined(stints[0]?.tyre, stints[0]?.compound, "medium"));
  initialTyre.value = Object.hasOwn(TYRES, startingTyre) ? startingTyre : "medium";
  pitStopRows.replaceChildren();
  for (const stint of stints.slice(1)) {
    const startLap = Math.max(2, Math.min(laps, Math.round(finiteNumber(firstDefined(stint.start_lap, stint.startLap), 2))));
    const tyre = String(firstDefined(stint.tyre, stint.compound, "medium"));
    createPitStopRow(startLap, Object.hasOwn(TYRES, tyre) ? tyre : "medium");
  }

  weatherRows.replaceChildren();
  const weather = [...plan.weather].sort((a, b) => finiteNumber(a.start_lap) - finiteNumber(b.start_lap));
  weather.forEach((phase, index) => {
    const condition = String(firstDefined(phase.condition, phase.weather, "dry"));
    const preset = WEATHER[condition] || WEATHER.dry;
    const startLap = index === 0 ? 1 : Math.max(2, Math.min(laps, Math.round(finiteNumber(firstDefined(phase.start_lap, phase.startLap), 2))));
    const temperature = finiteNumber(firstDefined(phase.track_temp_c, phase.trackTemperatureC), preset.temperature);
    const rawRain = finiteNumber(firstDefined(phase.rain_intensity, phase.rainIntensity), preset.rain / 100);
    const rainPercent = rawRain <= 1 ? rawRain * 100 : rawRain;
    createWeatherRow(startLap, Object.hasOwn(WEATHER, condition) ? condition : "dry", temperature, rainPercent, index > 0);
  });
  if (!weather.length) createWeatherRow(1, "dry", 25, 0, false);
  renderStrategyTimeline();
  clearError();
  navigateTo("run");
  form.scrollIntoView({ behavior: reducedMotion.matches ? "auto" : "smooth", block: "start" });
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  clearError();
  let plan;
  try {
    plan = collectPlan();
  } catch (error) {
    showError(error instanceof Error ? error.message : "Check the race strategy and try again.");
    return;
  }

  cancelAnimationFrame(replayFrame);
  replayRunning = false;
  setLoading(true);
  try {
    const data = await runSimulation(plan);
    presentSimulation(data, plan);
    navigateTo("results");
    void loadHistory({ quiet: true });
  } catch (error) {
    setSummarySkeleton(false);
    const message = error instanceof Error ? error.message : "The simulation could not be completed.";
    showError(`${message} Check that the local simulation server is running, then try again.`);
  } finally {
    setLoading(false);
  }
});

addPitStopButton.addEventListener("click", () => {
  clearError();
  const start = availableStart(".pit-start", Math.floor(currentLapCount() / 2) + 1);
  if (start === null) {
    showError("Every lap already has a scheduled tyre change. Remove a stop before adding another.");
    return;
  }
  createPitStopRow(start, "medium");
});

addWeatherButton.addEventListener("click", () => {
  clearError();
  const start = availableStart(".weather-start", Math.floor(currentLapCount() / 2) + 1);
  if (start === null) {
    showError("Every lap already has a weather phase. Remove a change before adding another.");
    return;
  }
  createWeatherRow(start, "damp", WEATHER.damp.temperature, WEATHER.damp.rain, true);
});

dryPresetButton.addEventListener("click", () => {
  clearError();
  applyPreset("dry");
});

rainPresetButton.addEventListener("click", () => {
  clearError();
  applyPreset("rain");
});

form.addEventListener("input", event => {
  if (event.target !== runButton) renderStrategyTimeline();
});

weatherRows.addEventListener("change", event => {
  if (event.target.classList.contains("weather-condition")) {
    const row = event.target.closest(".weather-row");
    const preset = WEATHER[event.target.value];
    if (preset) {
      row.querySelector(".track-temperature").value = String(preset.temperature);
      row.querySelector(".rain-intensity").value = String(preset.rain);
    }
  }
  renderStrategyTimeline();
});

circuitSelect.addEventListener("change", () => {
  clearError();
  updateCircuitContext();
});

window.addEventListener("hashchange", () => showView(viewFromHash()));
for (const link of navigationLinks) {
  link.addEventListener("click", () => {
    if (window.location.hash === link.getAttribute("href")) showView(link.dataset.viewTarget);
  });
}
clearHistoryButton.addEventListener("click", () => void clearHistory());
historyTableBody.addEventListener("click", event => {
  const target = event.target.closest("[data-history-id]");
  if (!target || !historyTableBody.contains(target)) return;
  void loadHistoryResult(target.dataset.historyId);
});
historyTableBody.addEventListener("keydown", event => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const row = event.target.closest("tr[data-history-id]");
  if (!row || event.target.closest("button")) return;
  event.preventDefault();
  void loadHistoryResult(row.dataset.historyId);
});
rerunSavedButton.addEventListener("click", loadSavedStrategyIntoBuilder);

replayButton.addEventListener("click", startReplay);
document.getElementById("zoomInButton").addEventListener("click", () => zoomAt(mapView.zoom * 1.5));
document.getElementById("zoomOutButton").addEventListener("click", () => zoomAt(mapView.zoom / 1.5));
document.getElementById("resetViewButton").addEventListener("click", resetMapView);
for (const input of Object.values(layerInputs)) input.addEventListener("change", () => renderTrack(replayProgress));

fullscreenButton.addEventListener("click", async () => {
  try {
    if (document.fullscreenElement === trackPanel) await document.exitFullscreen();
    else await trackPanel.requestFullscreen();
  } catch (error) {
    showError("This browser did not allow full-screen mode. You can still zoom and pan inside the large map.");
  }
});

document.addEventListener("fullscreenchange", () => {
  const active = document.fullscreenElement === trackPanel;
  fullscreenButton.textContent = active ? "Exit full screen" : "Full screen";
  fullscreenButton.setAttribute("aria-label", active ? "Exit circuit map full screen" : "Open circuit map full screen");
  requestAnimationFrame(() => renderTrack(replayProgress));
});

const resizeObserver = new ResizeObserver(() => {
  renderTrack(replayProgress);
  if (lapResults.classList.contains("is-visible")) {
    drawLapChart(chartLapTimes);
    drawSpeedTrace();
    if (tyreLapPoints.length) drawTyreWearChart(tyreLapPoints);
  }
});
resizeObserver.observe(trackCanvas);
resizeObserver.observe(chartCanvas);
if (speedTraceCanvas) resizeObserver.observe(speedTraceCanvas);
if (tyreWearCanvas) resizeObserver.observe(tyreWearCanvas);

reducedMotion.addEventListener("change", () => {
  if (reducedMotion.matches && replayRunning) {
    cancelAnimationFrame(replayFrame);
    replayRunning = false;
    replayProgress = 1;
    renderTrack(1);
    liveText.textContent = "Replay complete";
  }
});

function launchOptions() {
  const parameters = new URLSearchParams(window.location.search);
  const requestedLaps = Number(parameters.get("laps"));
  if (Number.isInteger(requestedLaps) && requestedLaps >= 1 && requestedLaps <= maximumLaps) {
    lapsInput.value = String(requestedLaps);
    document.getElementById("requestedLaps").textContent = `of ${requestedLaps} ${requestedLaps === 1 ? "lap" : "laps"}`;
  }
  const requestedCircuit = parameters.get("circuit_id") || parameters.get("circuit");
  if (requestedCircuit && circuitCatalog.some(circuit => circuit.id === requestedCircuit)) {
    circuitSelect.value = requestedCircuit;
    updateCircuitContext();
  }
  return parameters.get("autorun") === "1";
}

async function initializeApplication() {
  showView(viewFromHash());
  populateCircuitSelect([], "gb-1948");
  for (const control of form.elements) control.disabled = true;
  try {
    await loadConfiguration();
  } catch (error) {
    showError(`${error instanceof Error ? error.message : "The circuit catalogue could not be loaded."} Using the Silverstone fallback.`);
  } finally {
    for (const control of form.elements) control.disabled = false;
  }

  const shouldAutorun = launchOptions();
  initializeStrategy();
  renderTrack(0);
  void loadHistory();
  if (shouldAutorun) requestAnimationFrame(() => form.requestSubmit());
}

void initializeApplication();
