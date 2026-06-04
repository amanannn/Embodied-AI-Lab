const ids = {
  kp: document.querySelector("#kp"),
  ki: document.querySelector("#ki"),
  kd: document.querySelector("#kd"),
  setpoint: document.querySelector("#setpoint"),
  disturbance: document.querySelector("#disturbance"),
};

const outputs = {
  kp: document.querySelector("#kpValue"),
  ki: document.querySelector("#kiValue"),
  kd: document.querySelector("#kdValue"),
  setpoint: document.querySelector("#setpointValue"),
  disturbance: document.querySelector("#disturbanceValue"),
  steadyError: document.querySelector("#steadyError"),
  overshoot: document.querySelector("#overshoot"),
  settlingTime: document.querySelector("#settlingTime"),
  peakControl: document.querySelector("#peakControl"),
};

const canvas = document.querySelector("#trajectoryCanvas");
const ctx = canvas.getContext("2d");
const runButton = document.querySelector("#runButton");
const resetButton = document.querySelector("#resetButton");
const statusMessage = document.querySelector("#statusMessage");

function readParams() {
  return {
    kp: Number(ids.kp.value),
    ki: Number(ids.ki.value),
    kd: Number(ids.kd.value),
    setpoint: Number(ids.setpoint.value),
    disturbance_force: Number(ids.disturbance.value),
  };
}

function syncLabels() {
  for (const key of ["kp", "ki", "kd", "setpoint", "disturbance"]) {
    outputs[key].textContent = Number(ids[key].value).toFixed(1);
  }
}

function readInitialPayload() {
  const node = document.querySelector("#initialPayload");
  if (!node || node.textContent.includes("__PID_INITIAL_PAYLOAD__")) {
    return null;
  }
  try {
    return JSON.parse(node.textContent);
  } catch (error) {
    setStatus(`Initial payload failed: ${error.message}`, true);
    return null;
  }
}

function buildQuery(params) {
  const query = new URLSearchParams();
  query.set("kp", params.kp);
  query.set("ki", params.ki);
  query.set("kd", params.kd);
  query.set("setpoint", params.setpoint);
  query.set("disturbance_force", params.disturbance_force);
  return query.toString();
}

async function runSimulation() {
  syncLabels();
  setStatus("Running simulation...");
  drawLoadingState();
  runButton.disabled = true;
  try {
    const response = await fetch(`/api/simulate?${buildQuery(readParams())}`);
    if (!response.ok) {
      throw new Error(`Simulation failed: ${response.status}`);
    }
    const payload = await response.json();
    drawTrajectory(payload.samples);
    updateMetrics(payload.metrics);
    setStatus(`Updated ${payload.samples.length} samples from the Python PID model.`);
  } catch (error) {
    setStatus(error.message, true);
    drawLoadingState("Simulation failed. Check that the Python server is running.");
  } finally {
    runButton.disabled = false;
  }
}

function setStatus(message, isError = false) {
  statusMessage.textContent = message;
  statusMessage.classList.toggle("error", isError);
}

function updateMetrics(metrics) {
  outputs.steadyError.textContent = metrics.steady_state_error.toFixed(3);
  outputs.overshoot.textContent = `${metrics.overshoot_percent.toFixed(1)}%`;
  outputs.settlingTime.textContent = `${metrics.settling_time.toFixed(2)}s`;
  outputs.peakControl.textContent = metrics.peak_control.toFixed(2);
}

function valueRange(samples) {
  const values = samples.flatMap((sample) => [
    sample.position,
    sample.setpoint,
    sample.control / 10,
  ]);
  const min = Math.min(-0.2, ...values);
  const max = Math.max(1.2, ...values);
  const pad = Math.max(0.2, (max - min) * 0.12);
  return { min: min - pad, max: max + pad };
}

function drawTrajectory(samples) {
  const width = canvas.width;
  const height = canvas.height;
  const inset = 54;
  const range = valueRange(samples);
  const maxTime = samples[samples.length - 1].time;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#07100c";
  ctx.fillRect(0, 0, width, height);
  drawGrid(width, height, inset);

  const project = (sample, value) => {
    const x = inset + (sample.time / maxTime) * (width - inset * 1.5);
    const normalized = (value - range.min) / (range.max - range.min);
    const y = height - inset - normalized * (height - inset * 1.6);
    return [x, y];
  };

  drawSeries(samples, (sample) => sample.setpoint, project, "#8ff27a", 3);
  drawSeries(samples, (sample) => sample.position, project, "#4ecdc4", 4);
  drawSeries(samples, (sample) => sample.control / 10, project, "#ff7a59", 2);

  const disturbed = samples.find((sample) => sample.disturbance !== 0);
  if (disturbed) {
    const [x] = project(disturbed, range.min);
    ctx.strokeStyle = "rgba(255, 95, 109, 0.72)";
    ctx.setLineDash([8, 8]);
    ctx.beginPath();
    ctx.moveTo(x, 30);
    ctx.lineTo(x, height - inset);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#ff5f6d";
    ctx.fillText("disturbance", x + 10, 42);
  }
}

function drawLoadingState(message = "Waiting for simulation data...") {
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#07100c";
  ctx.fillRect(0, 0, width, height);
  drawGrid(width, height, 54);
  ctx.fillStyle = "rgba(237, 247, 236, 0.76)";
  ctx.font = "22px ui-monospace, monospace";
  ctx.fillText(message, 72, height / 2);
}

function drawGrid(width, height, inset) {
  ctx.strokeStyle = "rgba(237, 247, 236, 0.09)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i += 1) {
    const y = inset + i * ((height - inset * 1.6) / 5);
    ctx.beginPath();
    ctx.moveTo(inset, y);
    ctx.lineTo(width - inset / 2, y);
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(237, 247, 236, 0.62)";
  ctx.font = "16px ui-monospace, monospace";
  ctx.fillText("position / scaled control", inset, 30);
  ctx.fillText("time", width - 92, height - 18);
}

function drawSeries(samples, accessor, project, color, lineWidth) {
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  samples.forEach((sample, index) => {
    const [x, y] = project(sample, accessor(sample));
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
}

function resetControls() {
  ids.kp.value = 8.0;
  ids.ki.value = 1.2;
  ids.kd.value = 2.6;
  ids.setpoint.value = 1.0;
  ids.disturbance.value = -1.5;
  runSimulation();
}

for (const input of Object.values(ids)) {
  input.addEventListener("input", syncLabels);
  input.addEventListener("change", runSimulation);
}

runButton.addEventListener("click", runSimulation);
resetButton.addEventListener("click", resetControls);

syncLabels();
drawLoadingState();
const initialPayload = readInitialPayload();
if (initialPayload) {
  drawTrajectory(initialPayload.samples);
  updateMetrics(initialPayload.metrics);
  setStatus(`Loaded ${initialPayload.samples.length} samples from the Python PID model.`);
} else {
  runSimulation();
}
