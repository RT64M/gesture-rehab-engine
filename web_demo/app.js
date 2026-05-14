import { BrowserScorer } from "./browser_scorer.js";
import { BrowserThresholdManager, getChallengeZone } from "./browser_threshold.js";
import { extractFeatureVector, normalizeForDrawing } from "./browser_features.js";

const COPY = {
  en: {
    brandMark: "GS",
    brandEyebrow: "Local Demo",
    brandTitle: "Gesture Scoring",
    navAria: "Demo navigation",
    tabAria: "Demo pages",
    tabClassifierIndex: "01",
    tabClassifier: "Live Classifier",
    tabColdStartIndex: "02",
    tabColdStart: "Cold Start",
    tabInverseIndex: "03",
    tabInverse: "Inverse Score",
    selectedGesture: "Selected Gesture",
    challengeBand: "Challenge Band",
    softRegulation: "Soft regulation, not clipping.",
    heroEyebrow: "Ten gestures / descriptor distance / adaptive tau",
    heroTitle: "Cold-start scoring, shown as a process.",
    gesturesMetric: "Gestures",
    accuracyMetric: "Accuracy",
    testSamplesMetric: "Test Samples",
    pageOne: "Page 01",
    classifierTitle: "Live Ten-Gesture Classifier",
    classifierCopy:
      "The first page samples the camera once per second. Each sampled screenshot is shown below the live preview, analyzed with MediaPipe hand landmarks, converted into the same 26-dimensional feature vector, and scored with an explanation.",
    gestureSample: "Gesture sample",
    startCamera: "Start camera",
    stopCamera: "Stop camera",
    pauseCamera: "Pause",
    resumeCamera: "Resume",
    captureFrame: "Capture frame",
    liveCamera: "Live camera",
    cameraFeed: "Hand Landmark Feed",
    cameraIdle: "Camera is stopped. Start it to run live classification.",
    cameraLoading: "Loading MediaPipe hand landmarker...",
    cameraRunning: "Camera is running. Show one hand in view.",
    cameraNoHand: "No hand detected in the current frame.",
    cameraError: "Camera or MediaPipe failed to start.",
    noLivePrediction: "Waiting for live hand landmarks.",
    cameraUnavailable: "This browser cannot open a camera stream.",
    sampledFrame: "Sampled frame",
    lastSnapshot: "Previous-second snapshot",
    snapshotIdle: "No sampled frame yet.",
    snapshotCaptured: "Analyzed sampled frame at {time}.",
    samplingPaused: "Sampling is paused. Resume to analyze new snapshots.",
    chapterThree: "Chapter 3 evidence",
    gestureMap: "Gesture Map",
    gestureMapAlt: "Ten gesture landmark mapping grid",
    openAsset: "Open asset",
    nearestDescriptors: "Nearest descriptors",
    distanceRanking: "Distance Ranking",
    featureExplanation: "Feature explanation",
    dominantDeviations: "Dominant Deviations",
    explainScore: "Score explanation",
    currentScore: "Current score",
    lowScoreReason: "Main reason",
    improvementCue: "Try next",
    scoreReasonTemplate: "{feature} is {direction} by {sigma} sigma for {gesture}.",
    scoreOkTemplate: "The hand shape is close to {gesture}; keep the same form.",
    cueTemplate: "Bring {feature} closer to the target shape, then hold the gesture steady.",
    cueOk: "Hold the gesture steady and repeat the same form.",
    directions: {
      high: "too high",
      low: "too low",
      wide: "too open",
      narrow: "not open enough",
    },
    pageTwo: "Page 02",
    coldStartTitle: "Cold-Start Process",
    coldStartCopy:
      "The second page uses Palm as the fixed example gesture. The first adaptive score has no personal history, so it matches the direct score; later attempts change only after the algorithm updates the user model and threshold state.",
    presetProcess: "Cold-start scenario",
    attemptCursor: "Attempt cursor",
    playButton: "Play",
    pauseButton: "Pause",
    backButton: "Back",
    nextButton: "Next",
    stateVariables: "State variables",
    patientAttempt: "Patient attempt",
    standardTarget: "Standard target",
    adaptiveParameters: "Adaptive Parameters",
    roundUpdates: "Round updates",
    tauBaselineMomentum: "Tau, Baseline, Momentum",
    pageThree: "Page 03",
    inverseTitle: "Inverse Scoring",
    inverseCopy:
      "The final page reads the Palm cold-start history backwards. The visible score can stay in a narrow band while tau tightens, so inverse progress can rise far more than the score alone suggests.",
    rawVsInterpreted: "Raw vs interpreted",
    scoreInverseProgress: "Score and Inverse Progress",
    weightedSignal: "Weighted signal",
    progressComponents: "Progress Components",
    inverseInsight: "Inverse insight",
    inverseInsightTemplate: "From attempt 1 to {attempt}, adaptive score changed by {scoreDelta} points, but tau changed from {tauStart} to {tauNow}; inverse progress is {progress}.",
    chapterSix: "Chapter 6 reference",
    stateVariablesFigure: "State Variables",
    inverseFigureAlt: "Inverse progress state variables",
    predicted: "Predicted",
    target: "Target",
    recall: "Recall",
    bestDistance: "Best Distance",
    margin: "Margin",
    targetDistance: "Target Distance",
    attempt: "Attempt",
    score: "Score",
    zone: "Zone",
    distance: "Distance",
    tau: "Tau",
    blend: "Blend",
    round: "Round",
    baseline: "Baseline",
    momentum: "Momentum",
    rawScore: "Raw score",
    fixedScore: "Fixed raw score",
    adaptiveScore: "Adaptive score",
    pending: "Pending",
    displayScore: "Display score",
    challengeBandLegend: "Challenge band",
    scoreChartAria: "Score chart with challenge band",
    inverseProgress: "Inverse progress",
    inverseChartAria: "Raw score and inverse progress chart",
    pageTitle: "Gesture Scoring Engine Demo",
    sigmaUnit: "sigma",
    componentWeight: "weight",
    tauChangeTemplate: "tau {oldTau} to {newTau} / baseline {baseline} / momentum {momentum} / {reason}",
    contribution: "Contribution",
    roundUpdateFallback: "Round updates appear after every {roundSize} attempts.",
    demoLoadError: "Demo load error",
    unableToLoad: "Unable to load the local demo data.",
    loadInstruction: "Run <code>uv run python scripts/export_demo_data.py</code>, then open the page through a local static server.",
    scenarioFocus: "Scenario focus",
    classifierProfiles: {
      reference: "Reference descriptor",
      mild_offset: "Mild offset",
      near_boundary: "Near boundary",
      severe_offset: "Severe offset",
    },
    presets: {
      steady_recovery: {
        label: "Steady recovery",
        description: "The first score matches direct scoring; later attempts reflect the learned user model and threshold state.",
      },
      mild_stabilization: {
        label: "Mild stabilization",
        description: "A user starts closer to the descriptor, so direct and adaptive scores remain relatively aligned while the model blends in history.",
      },
      severe_start_recovery: {
        label: "Severe start, then recovery",
        description: "The starting hand shape is very far from the descriptor. Fixed scoring stays punishing, while adaptive scoring changes only after the algorithm records earlier attempts.",
      },
      fatigue_plateau: {
        label: "Fatigue and plateau",
        description: "A mid-session dip interrupts progress; adaptive scoring reacts through the learned baseline and momentum.",
      },
    },
    components: {
      strictness_progression: "Tau tightening",
      descriptor_recovery: "Descriptor recovery",
      baseline_margin: "Baseline margin",
      momentum_trend: "Momentum trend",
      challenge_zone_alignment: "Challenge-zone alignment",
    },
    zones: {
      too_hard: "Too hard",
      in_zone: "In zone",
      too_easy: "Too easy",
    },
  },
};

const lang = "en";
const text = COPY[lang];

const state = {
  demoData: null,
  runtimeConfig: null,
  scorer: null,
  visionTasks: null,
  gesture: null,
  coldStartGesture: "palm",
  classifierSample: "reference",
  preset: "severe_start_recovery",
  history: [],
  cursor: 1,
  playing: false,
  timer: null,
  camera: {
    running: false,
    stream: null,
    landmarker: null,
    interval: null,
    paused: false,
    latestResult: null,
    latestLandmarks: null,
  },
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function applyStaticCopy() {
  $$("[data-copy]").forEach((node) => {
    node.textContent = text[node.dataset.copy] ?? "";
  });
  $$("[data-copy-attr]").forEach((node) => {
    node.dataset.copyAttr.split(";").forEach((pair) => {
      const [attr, key] = pair.split(":");
      if (attr && key && text[key] != null) {
        node.setAttribute(attr, text[key]);
      }
    });
  });
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`);
  }
  return response.json();
}

function fmt(value, digits = 2) {
  if (!Number.isFinite(Number(value))) {
    return "0.00";
  }
  return Number(value).toFixed(digits);
}

function pct(value) {
  return `${Math.round(Number(value) * 100)}%`;
}

function labelGesture(gesture) {
  return state.runtimeConfig.gesture_labels[gesture] ?? gesture;
}

function featureLabel(name) {
  return name.replaceAll("_", " ");
}

function deviationDirection(feature, zScore) {
  if (feature.startsWith("spread_")) {
    return zScore > 0 ? text.directions.wide : text.directions.narrow;
  }
  return zScore > 0 ? text.directions.high : text.directions.low;
}

function zoneLabel(code) {
  return text.zones[code] ?? code.replaceAll("_", " ");
}

function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

function drawSkeleton(canvas, points, options = {}) {
  if (!canvas || !points?.length) {
    return;
  }
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(260, rect.width || canvas.clientWidth || 320);
  const height = Math.max(220, rect.height || canvas.clientHeight || 260);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = options.background ?? "rgba(255, 253, 247, 0.84)";
  context.fillRect(0, 0, width, height);
  const padding = 28;
  const toCanvas = ([x, y]) => [
    padding + x * (width - padding * 2),
    padding + y * (height - padding * 2),
  ];
  const connections = state.runtimeConfig.landmark_config.connections ?? [];
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = options.line ?? "#2d6f5f";
  context.lineWidth = options.lineWidth ?? 4;
  connections.forEach(([from, to]) => {
    if (!points[from] || !points[to]) {
      return;
    }
    const start = toCanvas(points[from]);
    const end = toCanvas(points[to]);
    context.beginPath();
    context.moveTo(start[0], start[1]);
    context.lineTo(end[0], end[1]);
    context.stroke();
  });
  points.forEach((point, index) => {
    const [x, y] = toCanvas(point);
    context.beginPath();
    context.fillStyle = index === 0 ? options.wrist ?? "#bc5e35" : options.dot ?? "#1e241f";
    context.arc(x, y, index === 0 ? 5 : 4, 0, Math.PI * 2);
    context.fill();
  });
}

function drawSnapshotFrame(result = null) {
  const video = $("#camera-video");
  const canvas = $("#snapshot-canvas");
  if (!video || !canvas || video.readyState < video.HAVE_CURRENT_DATA) {
    return;
  }
  const ratio = window.devicePixelRatio || 1;
  const width = video.videoWidth || 640;
  const height = video.videoHeight || 360;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.save();
  context.translate(width, 0);
  context.scale(-1, 1);
  context.drawImage(video, 0, 0, width, height);
  context.restore();

  const landmarks = result?.landmarks?.[0];
  if (!landmarks?.length) {
    return;
  }
  const connections = state.runtimeConfig.landmark_config.connections ?? [];
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = "rgba(255, 253, 247, 0.9)";
  context.lineWidth = Math.max(3, width / 180);
  connections.forEach(([from, to]) => {
    const start = landmarks[from];
    const end = landmarks[to];
    if (!start || !end) {
      return;
    }
    context.beginPath();
    context.moveTo((1 - start.x) * width, start.y * height);
    context.lineTo((1 - end.x) * width, end.y * height);
    context.stroke();
  });
  landmarks.forEach((point, index) => {
    context.beginPath();
    context.fillStyle = index === 0 ? "#bc5e35" : "#2d6f5f";
    context.arc((1 - point.x) * width, point.y * height, index === 0 ? 6 : 4.5, 0, Math.PI * 2);
    context.fill();
  });
}

function clearSnapshotFrame() {
  const canvas = $("#snapshot-canvas");
  if (!canvas) {
    return;
  }
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(320, rect.width || 640);
  const height = Math.max(180, rect.height || 360);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "rgba(30, 36, 31, 0.08)";
  context.fillRect(0, 0, width, height);
}

async function ensureHandLandmarker() {
  if (state.camera.landmarker) {
    return state.camera.landmarker;
  }
  const config = state.runtimeConfig.camera_classifier;
  if (!state.visionTasks) {
    state.visionTasks = await import(config.vision_module_url);
  }
  const { FilesetResolver, HandLandmarker } = state.visionTasks;
  const fileset = await FilesetResolver.forVisionTasks(config.vision_wasm_url);
  state.camera.landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: config.model_asset_url,
    },
    runningMode: "VIDEO",
    numHands: config.num_hands,
    minHandDetectionConfidence: config.min_hand_detection_confidence,
    minHandPresenceConfidence: config.min_hand_presence_confidence,
    minTrackingConfidence: config.min_tracking_confidence,
  });
  return state.camera.landmarker;
}

function processSnapshotResult(result) {
  drawSnapshotFrame(result);
  const landmarks = result?.landmarks?.[0];
  const timestamp = new Date().toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  $("#snapshot-status").textContent = text.snapshotCaptured.replace("{time}", timestamp);
  if (!landmarks?.length) {
    state.camera.latestResult = null;
    state.camera.latestLandmarks = null;
    $("#camera-status").textContent = text.cameraNoHand;
    return;
  }
  const vector = extractFeatureVector(landmarks, state.runtimeConfig.landmark_config);
  const drawingPoints = normalizeForDrawing(landmarks, state.runtimeConfig.landmark_config);
  state.camera.latestResult = { vector, drawingPoints };
  state.camera.latestLandmarks = landmarks;
  $("#camera-status").textContent = text.cameraRunning;
  renderClassifierResultFromVector(vector, drawingPoints);
}

function detectionFromSnapshotCanvas() {
  const canvas = $("#snapshot-canvas");
  if (!canvas || !state.camera.landmarker) {
    return null;
  }
  return state.camera.landmarker.detect(canvas);
}

function sampleCameraFrame() {
  if (!state.camera.running || state.camera.paused) {
    return;
  }
  const video = $("#camera-video");
  const landmarker = state.camera.landmarker;
  if (!landmarker || video.readyState < video.HAVE_CURRENT_DATA) {
    $("#camera-status").textContent = text.noLivePrediction;
    return;
  }
  drawSnapshotFrame(null);
  const result = detectionFromSnapshotCanvas();
  processSnapshotResult(result);
}

function startSampling() {
  if (state.camera.interval) {
    clearInterval(state.camera.interval);
  }
  state.camera.interval = setInterval(sampleCameraFrame, 1000);
  sampleCameraFrame();
}

function pauseSampling() {
  state.camera.paused = true;
  if (state.camera.interval) {
    clearInterval(state.camera.interval);
    state.camera.interval = null;
  }
  $("#camera-toggle").textContent = text.resumeCamera;
  $("#camera-status").textContent = text.samplingPaused;
  $("#snapshot-status").textContent = text.samplingPaused;
}

function resumeSampling() {
  state.camera.paused = false;
  $("#camera-toggle").textContent = text.pauseCamera;
  $("#camera-status").textContent = text.cameraRunning;
  startSampling();
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    $("#camera-status").textContent = text.cameraUnavailable;
    return;
  }
  $("#camera-status").textContent = text.cameraLoading;
  $("#camera-toggle").disabled = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    const video = $("#camera-video");
    state.camera.stream = stream;
    video.srcObject = stream;
    await video.play();
    const landmarker = await ensureHandLandmarker();
    state.camera.landmarker = landmarker;
    state.camera.running = true;
    state.camera.paused = false;
    $("#camera-toggle").textContent = text.pauseCamera;
    $("#camera-status").textContent = text.cameraRunning;
    startSampling();
  } catch (error) {
    stopCamera();
    $("#camera-status").textContent = `${text.cameraError} ${error.message ?? ""}`.trim();
  } finally {
    $("#camera-toggle").disabled = false;
  }
}

function stopCamera() {
  state.camera.running = false;
  state.camera.paused = false;
  if (state.camera.interval) {
    clearInterval(state.camera.interval);
    state.camera.interval = null;
  }
  if (state.camera.stream) {
    state.camera.stream.getTracks().forEach((track) => track.stop());
    state.camera.stream = null;
  }
  const video = $("#camera-video");
  if (video) {
    video.pause();
    video.srcObject = null;
  }
  state.camera.latestResult = null;
  state.camera.latestLandmarks = null;
  clearSnapshotFrame();
  $("#camera-toggle").textContent = text.startCamera;
  $("#camera-status").textContent = text.cameraIdle;
  $("#snapshot-status").textContent = text.snapshotIdle;
}

function captureFrame() {
  if (!state.camera.running || !state.camera.landmarker) {
    renderClassifier();
    return;
  }
  const video = $("#camera-video");
  if (video.readyState < video.HAVE_CURRENT_DATA) {
    $("#camera-status").textContent = text.noLivePrediction;
    return;
  }
  drawSnapshotFrame(null);
  const result = detectionFromSnapshotCanvas();
  processSnapshotResult(result);
}

function currentChallenge() {
  return state.runtimeConfig.cold_start_demo;
}

function stage2Clone() {
  return JSON.parse(JSON.stringify(state.runtimeConfig.stage2));
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function thresholdConfig() {
  const base = state.runtimeConfig.stage3.defaults;
  const cold = state.runtimeConfig.cold_start_demo;
  return {
    ...base,
    round_size: cold.round_size,
    tau_init: cold.tau_init,
    tau_min: cold.tau_min,
    tau_max: cold.tau_max,
    challenge_low: cold.challenge_low,
    challenge_high: cold.challenge_high,
  };
}

function vectorForStrength(gesture, strength, wobble, stepIndex) {
  const descriptor = state.runtimeConfig.stage2.descriptors[gesture];
  const samples = state.demoData.classifier_samples[gesture];
  const reference = samples.find((sample) => sample.id === "reference") ?? samples[0];
  const severe = samples.find((sample) => sample.id === "severe_offset") ?? samples[samples.length - 1];
  const direction = severe.vector.map((value, index) => value - reference.vector[index]);
  return descriptor.mu.map((value, index) => {
    const wave = Math.sin((index + 1) * (stepIndex + 2) * 0.61) * descriptor.sigma_diag[index] * wobble;
    return value + direction[index] * (strength / 1.45) + wave;
  });
}

function skeletonForStrength(gesture, strength, wobble, stepIndex) {
  const standard = state.demoData.gesture_skeletons[gesture]?.standard ?? [];
  if (!standard.length) {
    return [];
  }
  if (gesture === "palm") {
    const normalizedStrength = clamp(strength / 2.4, 0, 1);
    const palmCenter = [0.5, 0.66];
    return standard.map((point, index) => {
      if (index === 0) {
        return point;
      }
      const fingerIndex = Math.floor((index - 1) / 4);
      const tipProgress = index === 0 ? 0 : ((index - 1) % 4 + 1) / 4;
      const lateralBias = [-0.08, -0.035, 0, 0.035, 0.075][fingerIndex] ?? 0;
      const curledTarget = [
        palmCenter[0] + lateralBias * (1 - tipProgress * 0.35),
        palmCenter[1] + tipProgress * 0.2,
      ];
      const curlWeight = normalizedStrength * (0.35 + tipProgress * 0.65);
      const tremor = Math.sin((index + 1) * 0.8 + stepIndex * 0.25) * wobble * 0.08;
      return [
        clamp(point[0] + (curledTarget[0] - point[0]) * curlWeight + tremor, 0, 1),
        clamp(point[1] + (curledTarget[1] - point[1]) * curlWeight + Math.abs(tremor) * 0.3, 0, 1),
      ];
    });
  }
  const samples = state.demoData.classifier_samples[gesture];
  const competitor = samples?.[0]?.competitor_gesture;
  const competitorSkeleton = state.demoData.gesture_skeletons[competitor]?.standard ?? standard;
  return standard.map((point, index) => {
    const other = competitorSkeleton[index] ?? point;
    const wave = Math.sin((index + 1) * (stepIndex + 2) * 0.53) * wobble * 0.2;
    return [
      clamp(point[0] + (other[0] - point[0]) * (strength / 1.65) + wave, 0, 1),
      clamp(point[1] + (other[1] - point[1]) * (strength / 1.65) - wave * 0.6, 0, 1),
    ];
  });
}

function computeDescriptorRecovery(history, index) {
  if (!history.length) {
    return 0;
  }
  const start = history[0].mahalanobis_distance;
  const current = history[index].mahalanobis_distance;
  if (start <= 1e-6) {
    return current <= 1e-6 ? 1 : 0;
  }
  return clamp(1 - current / start, 0, 1);
}

function computeComponents(history, index) {
  const item = history[index];
  const initialTau = history[0]?.inverse_tau ?? currentChallenge().tau_init;
  const currentTau = item.inverse_tau ?? item.threshold_tau;
  const strictness = clamp((initialTau - currentTau) / Math.max(initialTau - currentChallenge().tau_min, 1e-6), 0, 1);
  const descriptorRecovery = computeDescriptorRecovery(history, index);
  const baselineMargin = clamp((item.display_score - item.baseline) / 25, 0, 1);
  const momentum = clamp(item.momentum / 10, 0, 1);
  const zone = item.challenge_zone === "in_zone" ? 1 : item.challenge_zone === "too_easy" ? 0.5 : 0;
  return {
    strictness_progression: strictness,
    descriptor_recovery: descriptorRecovery,
    baseline_margin: baselineMargin,
    momentum_trend: momentum,
    challenge_zone_alignment: zone,
  };
}

function inverseProgress(history, index) {
  const components = computeComponents(history, index);
  const weights = {
    strictness_progression: 0.55,
    descriptor_recovery: 0.25,
    baseline_margin: 0.08,
    momentum_trend: 0.07,
    challenge_zone_alignment: 0.05,
  };
  return Object.entries(weights).reduce((sum, [name, weight]) => sum + components[name] * weight, 0);
}

function inverseTauForIndex(index, total) {
  const cold = currentChallenge();
  const progress = index / Math.max(total - 1, 1);
  return cold.tau_init - (cold.tau_init - cold.tau_min) * (progress ** 0.85) * 0.82;
}

function buildColdStartHistory() {
  const preset = state.demoData.cold_start_presets.find((item) => item.id === state.preset);
  const adaptiveScorer = new BrowserScorer({ ...state.runtimeConfig, stage2: stage2Clone() });
  const fixedScorer = new BrowserScorer({ ...state.runtimeConfig, stage2: deepClone(state.runtimeConfig.stage2) });
  const threshold = new BrowserThresholdManager(thresholdConfig());
  const rows = [];
  const gesture = state.coldStartGesture;

  preset.strengths.forEach((strength, index) => {
    const vector = vectorForStrength(gesture, strength, preset.wobble, index);
    const fixedResult = fixedScorer.score(vector, gesture, {
      updateUserModel: false,
    });
    const thresholdState = threshold.getState(gesture);
    const result = adaptiveScorer.score(vector, gesture, {
      updateUserModel: true,
      tauOverride: thresholdState.tau,
    });
    const adaptiveDisplayScore = result.display_score;
    threshold.registerAttempt(gesture, adaptiveDisplayScore, result.raw_score);
    let roundUpdate = null;
    if (threshold.hasFullRound(gesture)) {
      roundUpdate = threshold.finalizeRound(gesture);
    }
    const nextState = threshold.getState(gesture);
    rows.push({
      ...result,
      attempt: index + 1,
      preset_id: preset.id,
      preset_label: text.presets[preset.id].label,
      preset_description: text.presets[preset.id].description,
      strength,
      fixed_score: fixedResult.display_score,
      fixed_raw_score: fixedResult.raw_score,
      fixed_distance: fixedResult.mahalanobis_distance,
      threshold_tau: nextState.tau,
      inverse_tau: inverseTauForIndex(index, preset.strengths.length),
      tau_override: thresholdState.tau,
      baseline: nextState.baseline,
      momentum: nextState.momentum,
      round_index: nextState.round_index,
      pending_attempts: nextState.pending_attempts,
      challenge_zone: getChallengeZone(adaptiveDisplayScore, currentChallenge().challenge_low, currentChallenge().challenge_high),
      fixed_challenge_zone: getChallengeZone(fixedResult.display_score, currentChallenge().challenge_low, currentChallenge().challenge_high),
      round_update: roundUpdate,
      vector,
      patient_skeleton: skeletonForStrength(gesture, strength, preset.wobble, index),
      standard_skeleton: state.demoData.gesture_skeletons[gesture]?.standard ?? [],
    });
  });

  rows.forEach((row, index) => {
    row.components = computeComponents(rows, index);
    row.inverse_progress = inverseProgress(rows, index);
  });
  return rows;
}

function populateSelectors() {
  const gestures = state.demoData.phase1_summary.gestures;
  const gestureOptions = gestures.map((gesture) => `<option value="${gesture}">${labelGesture(gesture)}</option>`).join("");
  $("#global-gesture-select").innerHTML = gestureOptions;

  $("#preset-select").innerHTML = state.demoData.cold_start_presets
    .map((preset) => `<option value="${preset.id}">${text.presets[preset.id].label}</option>`)
    .join("");
  $("#preset-select").value = state.preset;
}

function setGesture(gesture) {
  state.gesture = gesture;
  $("#global-gesture-select").value = gesture;
  rebuildHistory();
  renderAll();
}

function setPreset(preset) {
  state.preset = preset;
  $("#preset-select").value = preset;
  rebuildHistory();
  renderColdStart();
  renderInverse();
}

function rebuildHistory() {
  state.history = buildColdStartHistory();
  state.cursor = Math.min(state.cursor, state.history.length);
  $("#attempt-range").max = `${state.history.length}`;
  $("#attempt-range").value = `${state.cursor}`;
}

function renderHero() {
  $("#hero-title").textContent = text.heroTitle;
  const summary = state.demoData.phase1_summary;
  const metrics = [
    [text.gesturesMetric, summary.gestures.length],
    [text.accuracyMetric, `${summary.overall_accuracy}%`],
    [text.testSamplesMetric, summary.total_test_samples.toLocaleString("en-US")],
  ];
  $("#hero-metrics").innerHTML = metrics
    .map(
      ([label, value]) => `
        <div class="metric-tile">
          <strong>${value}</strong>
          <span>${label}</span>
        </div>
      `,
    )
    .join("");
  const challenge = currentChallenge();
  $("#rail-zone-range").textContent = `${challenge.challenge_low}-${challenge.challenge_high}`;
}

function renderClassifierResultFromVector(vector, drawingPoints = null) {
  const predicted = state.scorer.classify(vector);
  const predictedGesture = predicted.predictedGesture;
  const distances = Object.entries(predicted.populationDistances)
    .map(([gesture, distance]) => ({ gesture, distance }))
    .sort((left, right) => left.distance - right.distance);
  const targetGesture = distances[0]?.gesture ?? predictedGesture;
  const scored = state.scorer.score(vector, targetGesture, {
    updateUserModel: false,
    topKFeedback: 5,
  });
  const targetScored = state.scorer.score(vector, state.gesture, {
    updateUserModel: false,
    topKFeedback: 5,
  });
  const best = distances[0];
  const second = distances[1] ?? best;
  const margin = second ? second.distance - best.distance : 0;

  $("#classifier-result").innerHTML = [
    [text.predicted, labelGesture(predictedGesture), "emphasis"],
    [text.bestDistance, fmt(best?.distance ?? 0, 2), ""],
    [text.margin, fmt(margin, 2), ""],
    [text.target, labelGesture(state.gesture), ""],
    [text.recall, `${state.demoData.phase1_summary.recalls[predictedGesture] ?? 0}%`, ""],
    [text.distance, fmt(predicted.populationDistances[state.gesture] ?? best?.distance ?? 0, 2), ""],
  ]
    .map(
      ([label, value, extra]) => `
        <div class="result-tile ${extra}">
          <strong>${value}</strong>
          <span>${label}</span>
        </div>
      `,
    )
    .join("");

  const maxDistance = Math.max(...distances.slice(0, 5).map((item) => item.distance), 1);
  $("#distance-ranking").innerHTML = distances
    .slice(0, 5)
    .map((item, index) => {
      const closeness = 1 - item.distance / maxDistance;
      const width = Math.max(8, closeness * 100);
      return `
        <div class="ranking-row">
          <div class="ranking-top">
            <strong>${index + 1}. ${labelGesture(item.gesture)}</strong>
            <span>${fmt(item.distance, 2)}</span>
          </div>
          <div class="bar-track"><div class="bar-fill ${item.gesture === predictedGesture ? "" : "rust"}" style="width:${width}%"></div></div>
        </div>
      `;
    })
    .join("");

  const maxDeviation = Math.max(...scored.feature_deviations.map((item) => item.magnitude), 1);
  $("#classifier-deviations").innerHTML = scored.feature_deviations
    .slice(0, 5)
    .map((item) => {
      const width = Math.max(8, (item.magnitude / maxDeviation) * 100);
      return `
        <div class="feature-row">
          <div class="feature-top">
            <strong>${featureLabel(item.feature)}</strong>
            <span>${fmt(item.z_score, 2)} ${text.sigmaUnit}</span>
          </div>
          <div class="bar-track"><div class="bar-fill blue" style="width:${width}%"></div></div>
        </div>
      `;
    })
    .join("");

  const topDeviation = targetScored.feature_deviations[0];
  const explanation = topDeviation && topDeviation.magnitude >= 0.6
    ? {
        reason: text.scoreReasonTemplate
          .replace("{feature}", featureLabel(topDeviation.feature))
          .replace("{direction}", deviationDirection(topDeviation.feature, topDeviation.z_score))
          .replace("{sigma}", fmt(Math.abs(topDeviation.z_score), 1))
          .replace("{gesture}", labelGesture(state.gesture)),
        cue: text.cueTemplate.replace("{feature}", featureLabel(topDeviation.feature)),
      }
    : {
        reason: text.scoreOkTemplate.replace("{gesture}", labelGesture(state.gesture)),
        cue: text.cueOk,
      };
  $("#score-explanation").innerHTML = `
    <p class="eyebrow">${text.explainScore}</p>
    <div class="explanation-grid">
      <div class="explanation-tile emphasis">
        <strong>${targetScored.display_score}</strong>
        <span>${text.currentScore}</span>
      </div>
      <div class="explanation-tile">
        <strong>${text.lowScoreReason}</strong>
        <span>${explanation.reason}</span>
      </div>
      <div class="explanation-tile">
        <strong>${text.improvementCue}</strong>
        <span>${explanation.cue}</span>
      </div>
    </div>
  `;

  void drawingPoints;
}

function renderClassifier() {
  $("#classifier-copy").textContent = text.classifierCopy;
  const samples = state.demoData.classifier_samples[state.gesture];
  const sample = samples.find((item) => item.id === state.classifierSample) ?? samples[0];
  const descriptor = state.runtimeConfig.stage2.descriptors[state.gesture];
  $("#camera-status").textContent = state.camera.running ? text.cameraRunning : text.cameraIdle;
  $("#camera-toggle").textContent = state.camera.running
    ? (state.camera.paused ? text.resumeCamera : text.pauseCamera)
    : text.startCamera;
  $("#camera-snapshot").textContent = text.captureFrame;
  $("#snapshot-status").textContent ||= text.snapshotIdle;

  if (state.camera.latestResult) {
    renderClassifierResultFromVector(state.camera.latestResult.vector, state.camera.latestResult.drawingPoints);
  } else {
    renderClassifierResultFromVector(sample.vector);
  }

  if (descriptor) {
    document.title = text.pageTitle;
  }
}

function renderSkeletonComparison(current) {
  drawSkeleton($("#patient-skeleton"), current.patient_skeleton, {
    line: "#bc5e35",
    dot: "#1e241f",
    wrist: "#a84b42",
    background: "rgba(255, 253, 247, 0.9)",
  });
  drawSkeleton($("#standard-skeleton"), current.standard_skeleton, {
    line: "#2d6f5f",
    dot: "#1e241f",
    wrist: "#315f89",
    background: "rgba(255, 253, 247, 0.9)",
  });
}

function chartPath(values, xScale, yScale) {
  return values.map((value, index) => `${index === 0 ? "M" : "L"} ${xScale(index)} ${yScale(value)}`).join(" ");
}

function renderScoreBandChart() {
  const rows = state.history;
  const upto = rows.slice(0, state.cursor);
  const width = 760;
  const height = 320;
  const pad = { left: 42, right: 18, top: 22, bottom: 38 };
  const xScale = (index) => pad.left + (index / Math.max(rows.length - 1, 1)) * (width - pad.left - pad.right);
  const yScale = (value) => height - pad.bottom - (value / 100) * (height - pad.top - pad.bottom);
  const challenge = currentChallenge();
  const zoneY1 = yScale(challenge.challenge_high);
  const zoneY2 = yScale(challenge.challenge_low);
  const scoreValues = upto.map((item) => item.display_score);
  const fixedValues = upto.map((item) => item.fixed_score);
  const baselineValues = upto.map((item) => item.baseline);
  const scorePath = chartPath(scoreValues, xScale, yScale);
  const fixedPath = chartPath(fixedValues, xScale, yScale);
  const baselinePath = chartPath(baselineValues, xScale, yScale);
  const points = upto
    .map(
      (item, index) => `
        <circle cx="${xScale(index)}" cy="${yScale(item.display_score)}" r="${index + 1 === state.cursor ? 6 : 4}" class="point ${item.challenge_zone}"></circle>
      `,
    )
    .join("");

  $("#score-band-chart").innerHTML = `
    <div class="chart-shell">
      <svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${text.scoreChartAria}">
        <rect class="zone-band" x="${pad.left}" y="${zoneY1}" width="${width - pad.left - pad.right}" height="${zoneY2 - zoneY1}"></rect>
        <line class="grid-line" x1="${pad.left}" x2="${width - pad.right}" y1="${yScale(85)}" y2="${yScale(85)}"></line>
        <line class="grid-line" x1="${pad.left}" x2="${width - pad.right}" y1="${yScale(60)}" y2="${yScale(60)}"></line>
        <line class="axis-line" x1="${pad.left}" x2="${pad.left}" y1="${pad.top}" y2="${height - pad.bottom}"></line>
        <line class="axis-line" x1="${pad.left}" x2="${width - pad.right}" y1="${height - pad.bottom}" y2="${height - pad.bottom}"></line>
        <path d="${baselinePath}" fill="none" stroke="#315f89" stroke-width="2.5"></path>
        <path d="${fixedPath}" fill="none" stroke="#8b8f89" stroke-width="3" stroke-dasharray="7 6"></path>
        <path d="${scorePath}" fill="none" stroke="#bc5e35" stroke-width="3.5"></path>
        ${points}
        <text x="${pad.left + 6}" y="${zoneY1 - 6}" class="chart-label">85</text>
        <text x="${pad.left + 6}" y="${zoneY2 + 16}" class="chart-label">60</text>
      </svg>
      <div class="chart-legend">
        <span class="legend-item"><i class="legend-swatch" style="background:#8b8f89"></i>${text.fixedScore}</span>
        <span class="legend-item"><i class="legend-swatch" style="background:#bc5e35"></i>${text.adaptiveScore}</span>
        <span class="legend-item"><i class="legend-swatch" style="background:#315f89"></i>${text.baseline}</span>
        <span class="legend-item"><i class="legend-swatch" style="background:rgba(45,111,95,.24)"></i>${text.challengeBandLegend}</span>
      </div>
    </div>
  `;
}

function renderColdStart() {
  $("#cold-start-copy").textContent = text.coldStartCopy;
  const rows = state.history;
  const current = rows[state.cursor - 1];
  const presetCopy = text.presets[current.preset_id];
  $("#scenario-description").innerHTML = `<strong>${text.scenarioFocus}:</strong> ${presetCopy.description}`;
  $("#attempt-range").value = `${state.cursor}`;
  $("#play-toggle").textContent = state.playing ? text.pauseButton : text.playButton;

  $("#attempt-summary").innerHTML = [
    [text.attempt, `${current.attempt}/${rows.length}`, "emphasis"],
    [text.fixedScore, current.fixed_score, ""],
    [text.adaptiveScore, current.display_score, ""],
    [text.zone, zoneLabel(current.challenge_zone), ""],
    [text.distance, fmt(current.fixed_distance, 2), ""],
    [text.tau, fmt(current.inverse_tau, 3), ""],
    [text.blend, fmt(current.blend_weight, 2), ""],
  ]
    .map(
      ([label, value, extra]) => `
        <div class="result-tile ${extra}">
          <strong>${value}</strong>
          <span>${label}</span>
        </div>
      `,
    )
    .join("");

  $("#state-grid").innerHTML = [
    [text.round, current.round_index || 0],
    [text.baseline, fmt(current.baseline, 2)],
    [text.momentum, fmt(current.momentum, 2)],
    [text.rawScore, fmt(current.fixed_raw_score, 3)],
    [text.predicted, labelGesture(current.predicted_gesture)],
    [text.pending, `${current.pending_attempts}/${currentChallenge().round_size}`],
  ]
    .map(
      ([label, value]) => `
        <div class="state-card">
          <strong>${value}</strong>
          <span>${label}</span>
        </div>
      `,
    )
    .join("");

  renderSkeletonComparison(current);

  const updates = rows.filter((item) => item.round_update);
  $("#round-timeline").innerHTML = updates.length
    ? updates
        .map((item) => {
          const update = item.round_update;
          return `
            <div class="timeline-row ${item.attempt <= current.attempt ? "active" : ""}">
              <div class="timeline-top">
                <strong>${text.round} ${update.round_index}</strong>
                <span class="zone-chip ${update.challenge_zone}">${zoneLabel(update.challenge_zone)}</span>
              </div>
              <div class="timeline-detail">
                ${text.tauChangeTemplate
                  .replace("{oldTau}", fmt(update.tau_old, 3))
                  .replace("{newTau}", fmt(update.tau_new, 3))
                  .replace("{baseline}", fmt(update.baseline, 2))
                  .replace("{momentum}", fmt(update.momentum, 2))
                  .replace("{reason}", update.reason.replaceAll("_", " "))}
              </div>
            </div>
          `;
        })
        .join("")
    : `<div class="timeline-detail">${text.roundUpdateFallback.replace("{roundSize}", currentChallenge().round_size)}</div>`;

  renderScoreBandChart();
}

function renderInverseChart() {
  const rows = state.history.slice(0, state.cursor);
  const width = 760;
  const height = 320;
  const pad = { left: 42, right: 18, top: 22, bottom: 38 };
  const xScale = (index) => pad.left + (index / Math.max(state.history.length - 1, 1)) * (width - pad.left - pad.right);
  const yScale = (value) => height - pad.bottom - (value / 100) * (height - pad.top - pad.bottom);
  const scoreValues = rows.map((item) => item.display_score);
  const fixedValues = rows.map((item) => item.fixed_score);
  const inverseValues = rows.map((item) => item.inverse_progress * 100);

  $("#inverse-chart").innerHTML = `
    <div class="chart-shell">
      <svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${text.inverseChartAria}">
        <line class="axis-line" x1="${pad.left}" x2="${pad.left}" y1="${pad.top}" y2="${height - pad.bottom}"></line>
        <line class="axis-line" x1="${pad.left}" x2="${width - pad.right}" y1="${height - pad.bottom}" y2="${height - pad.bottom}"></line>
        <path d="${chartPath(fixedValues, xScale, yScale)}" fill="none" stroke="#8b8f89" stroke-width="3" stroke-dasharray="7 6"></path>
        <path d="${chartPath(scoreValues, xScale, yScale)}" fill="none" stroke="#bc5e35" stroke-width="3.5"></path>
        <path d="${chartPath(inverseValues, xScale, yScale)}" fill="none" stroke="#2d6f5f" stroke-width="3.5"></path>
      </svg>
      <div class="chart-legend">
        <span class="legend-item"><i class="legend-swatch" style="background:#8b8f89"></i>${text.fixedScore}</span>
        <span class="legend-item"><i class="legend-swatch" style="background:#bc5e35"></i>${text.adaptiveScore}</span>
        <span class="legend-item"><i class="legend-swatch" style="background:#2d6f5f"></i>${text.inverseProgress}</span>
      </div>
    </div>
  `;
}

function renderInverse() {
  $("#inverse-copy").textContent = text.inverseCopy;
  const current = state.history[state.cursor - 1];
  const first = state.history[0];
  const components = current.components;
  const weights = {
    strictness_progression: 0.55,
    descriptor_recovery: 0.25,
    baseline_margin: 0.08,
    momentum_trend: 0.07,
    challenge_zone_alignment: 0.05,
  };
  $("#inverse-insight").innerHTML = `<strong>${text.inverseInsight}:</strong> ${text.inverseInsightTemplate
    .replace("{attempt}", current.attempt)
    .replace("{scoreDelta}", `${current.display_score - first.display_score >= 0 ? "+" : ""}${current.display_score - first.display_score}`)
    .replace("{tauStart}", fmt(first.inverse_tau, 3))
    .replace("{tauNow}", fmt(current.inverse_tau, 3))
    .replace("{progress}", pct(current.inverse_progress))}`;

  $("#inverse-headline").innerHTML = [
    [text.inverseProgress, pct(current.inverse_progress), "emphasis"],
    [text.fixedScore, current.fixed_score, ""],
    [text.adaptiveScore, current.display_score, ""],
    [text.tau, fmt(current.inverse_tau, 3), ""],
  ]
    .map(
      ([label, value, extra]) => `
        <div class="result-tile ${extra}">
          <strong>${value}</strong>
          <span>${label}</span>
        </div>
      `,
    )
    .join("");

  $("#component-stack").innerHTML = Object.entries(components)
    .map(([name, value]) => {
      const weight = weights[name];
      return `
        <div class="component-row">
          <div class="component-top">
            <strong>${text.components[name]}</strong>
            <span>${pct(value)} / ${text.componentWeight} ${fmt(weight, 2)}</span>
          </div>
          <div class="bar-track"><div class="bar-fill gold" style="width:${Math.max(5, value * 100)}%"></div></div>
          <div class="component-note">${text.contribution}: ${pct(value * weight)}</div>
        </div>
      `;
    })
    .join("");

  renderInverseChart();
}

function renderAll() {
  renderHero();
  renderClassifier();
  renderColdStart();
  renderInverse();
}

function showView(name) {
  $$(".tab-button").forEach((button) => button.classList.toggle("active", button.dataset.view === name));
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
  $(".gesture-card")?.classList.toggle("hidden", name !== "classifier");
  if (name === "classifier") {
    renderClassifier();
  }
  if (name === "cold-start") {
    renderColdStart();
  }
  if (name === "inverse") {
    if (state.cursor < 12) {
      state.cursor = Math.min(12, state.history.length);
      $("#attempt-range").value = `${state.cursor}`;
    }
    renderInverse();
  }
}

function stopPlayback() {
  state.playing = false;
  if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
  $("#play-toggle").textContent = text.playButton;
}

function startPlayback() {
  state.playing = true;
  $("#play-toggle").textContent = text.pauseButton;
  state.timer = setInterval(() => {
    if (state.cursor >= state.history.length) {
      stopPlayback();
      return;
    }
    state.cursor += 1;
    $("#attempt-range").value = `${state.cursor}`;
    renderColdStart();
    renderInverse();
  }, 850);
}

function bindEvents() {
  $$(".tab-button").forEach((button) => {
    button.addEventListener("click", () => showView(button.dataset.view));
  });

  $("#global-gesture-select").addEventListener("change", (event) => {
    setGesture(event.target.value);
  });

  $("#preset-select").addEventListener("change", (event) => {
    stopPlayback();
    state.cursor = 1;
    setPreset(event.target.value);
  });

  $("#attempt-range").addEventListener("input", (event) => {
    stopPlayback();
    state.cursor = Number(event.target.value);
    renderColdStart();
    renderInverse();
  });

  $("#step-back").addEventListener("click", () => {
    stopPlayback();
    state.cursor = Math.max(1, state.cursor - 1);
    renderColdStart();
    renderInverse();
  });

  $("#step-forward").addEventListener("click", () => {
    stopPlayback();
    state.cursor = Math.min(state.history.length, state.cursor + 1);
    renderColdStart();
    renderInverse();
  });

  $("#play-toggle").addEventListener("click", () => {
    if (state.playing) {
      stopPlayback();
      return;
    }
    if (state.cursor >= state.history.length) {
      state.cursor = 1;
    }
    startPlayback();
  });

  $("#camera-toggle").addEventListener("click", () => {
    if (state.camera.running) {
      if (state.camera.paused) {
        resumeSampling();
      } else {
        pauseSampling();
      }
      renderClassifier();
      return;
    }
    startCamera();
  });

  $("#camera-snapshot").addEventListener("click", captureFrame);

  window.addEventListener("beforeunload", stopCamera);
}

async function init() {
  const [demoData, runtimeConfig] = await Promise.all([
    fetchJson("./demo_data.json"),
    fetchJson("./runtime_config.json"),
  ]);

  state.demoData = demoData;
  state.runtimeConfig = runtimeConfig;
  state.scorer = new BrowserScorer(runtimeConfig);
  state.gesture = demoData.phase1_summary.gestures[0];
  state.coldStartGesture = demoData.phase1_summary.gestures.includes("palm")
    ? "palm"
    : demoData.phase1_summary.gestures[0];

  applyStaticCopy();
  populateSelectors();
  rebuildHistory();
  bindEvents();
  renderAll();
}

init().catch((error) => {
  document.body.innerHTML = `
    <main class="workspace">
      <section class="surface">
        <p class="eyebrow">${text.demoLoadError}</p>
        <h2>${text.unableToLoad}</h2>
        <p class="lead-copy">${error.message}</p>
        <p class="lead-copy">${text.loadInstruction}</p>
      </section>
    </main>
  `;
});
