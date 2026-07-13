"use strict";

const TYRES = {
  soft: { label: "Soft (C3)", short: "S", colour: "#ff4057" },
  medium: { label: "Medium (C2)", short: "M", colour: "#ffd166" },
  hard: { label: "Hard (C1)", short: "H", colour: "#e9edf5" },
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
let rowSequence = 0;
let tooltipPinned = false;
let hoveredTelemetryIndex = -1;
let lastGeometry = null;

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

function numericArray(value) {
  if (!Array.isArray(value)) return [];
  return value.map(item => Number(item)).filter(Number.isFinite);
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
    speed: payload.trajectory.speed_kph
  });
  const telemetry = pairedSeries(payload.telemetry.x_m, payload.telemetry.y_m, {
    time: payload.telemetry.time_s,
    lap: payload.telemetry.lap_number,
    speed: payload.telemetry.speed_kph,
    clearance: payload.telemetry.clearance_m || payload.telemetry.vehicle_edge_clearance_m
  }, {
    tyre: payload.telemetry.tyre || payload.telemetry.tyre_compound,
    weather: payload.telemetry.weather || payload.telemetry.weather_condition
  });

  if (track.x.length < 3) throw new Error("The server did not return enough circuit points to draw the track.");
  if (trajectory.x.length < 2) throw new Error("The server did not return enough racing-line points.");

  const rawSummary = payload.summary;
  return {
    track: {
      name: typeof payload.track.name === "string" ? payload.track.name : "Silverstone Circuit",
      ...track
    },
    trajectory,
    telemetry,
    summary: {
      requestedLaps: Math.max(0, Math.round(finiteNumber(rawSummary.requested_laps))),
      completedLaps: Math.max(0, Math.round(finiteNumber(rawSummary.completed_laps))),
      lapTimes: numericArray(rawSummary.lap_times_s),
      totalTime: finiteNumber(rawSummary.total_time_s, NaN),
      fastestLap: finiteNumber(rawSummary.fastest_lap_s, NaN),
      averageLap: finiteNumber(rawSummary.average_lap_s, NaN),
      offTrackEvents: Math.max(0, Math.round(finiteNumber(rawSummary.off_track_events))),
      maxSpeed: finiteNumber(rawSummary.max_speed_kph, NaN),
      meanSpeed: finiteNumber(rawSummary.mean_speed_kph, NaN),
      clearance: finiteNumber(rawSummary.minimum_vehicle_edge_clearance_m, NaN),
      terminationReason: typeof rawSummary.termination_reason === "string" ? rawSummary.termination_reason : "Unknown",
      strategy: payload.strategy || rawSummary.strategy || null,
      pitStops: Math.max(0, Math.round(finiteNumber(rawSummary.pit_stops))),
      pitStopTime: Math.max(0, finiteNumber(rawSummary.pit_stop_time_s))
    }
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

function populateLapTable(lapTimes) {
  const body = document.getElementById("lapTableBody");
  body.replaceChildren();
  if (!lapTimes.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 3;
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
    lapCell.textContent = `Lap ${index + 1}`;
    const timeCell = document.createElement("td");
    timeCell.textContent = formatDuration(time);
    const deltaCell = document.createElement("td");
    const delta = time - fastest;
    deltaCell.textContent = isFastest ? "FASTEST" : `+${delta.toFixed(3)} s`;
    deltaCell.className = isFastest ? "delta-zero" : "delta-positive";
    row.append(lapCell, timeCell, deltaCell);
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

  const lapInput = createNumberInput("pit-start", startLap, 2, MAX_LAPS, 1, "New tyre starts on lap");
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
    const lapInput = createNumberInput("weather-start", startLap, 2, MAX_LAPS, 1, "Weather phase starts on lap");
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
  return Number.isInteger(laps) && laps >= 1 && laps <= MAX_LAPS ? laps : 10;
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
  const laps = Number(lapsInput.value);
  if (!Number.isInteger(laps) || laps < 1 || laps > MAX_LAPS) {
    throw new Error(`Enter a whole number from 1 to ${MAX_LAPS} laps.`);
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

function presentSimulation(data, plan) {
  simulationData = data;
  lastPlan = plan;
  chartLapTimes = data.summary.lapTimes;
  document.getElementById("trackTitle").textContent = data.track.name;
  populateSummary(data.summary);
  populateLapTable(chartLapTimes);
  populateExecutedPlan(plan, data.summary);
  emptyState.classList.add("is-hidden");
  mapOverlay.classList.add("is-visible");
  replayButton.classList.add("is-visible");
  liveBadge.classList.add("is-visible");
  lapResults.classList.add("is-visible");
  resetMapView();
  drawLapChart(chartLapTimes);
  startReplay();
}

async function runSimulation(plan) {
  const response = await fetch("/api/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Accept": "application/json" },
    body: JSON.stringify(plan)
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
  if (lapResults.classList.contains("is-visible")) drawLapChart(chartLapTimes);
});
resizeObserver.observe(trackCanvas);
resizeObserver.observe(chartCanvas);

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
  if (Number.isInteger(requestedLaps) && requestedLaps >= 1 && requestedLaps <= MAX_LAPS) {
    lapsInput.value = String(requestedLaps);
    document.getElementById("requestedLaps").textContent = `of ${requestedLaps} ${requestedLaps === 1 ? "lap" : "laps"}`;
  }
  return parameters.get("autorun") === "1";
}

const shouldAutorun = launchOptions();
initializeStrategy();
renderTrack(0);
if (shouldAutorun) requestAnimationFrame(() => form.requestSubmit());
