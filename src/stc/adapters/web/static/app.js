import { STRINGS } from "./i18n.js";

const MAX_CHARS = 512;
const THEME_KEY = "stc.theme";
const LANG_KEY = "stc.lang";

const byId = (id) => document.getElementById(id);

const elements = {
  html: document.documentElement,
  themeToggle: byId("theme-toggle"),
  langToggle: byId("lang-toggle"),
  modelChip: byId("model-chip"),
  offlineBanner: byId("offline-banner"),
  form: byId("classify-form"),
  input: byId("text-input"),
  counter: byId("input-counter"),
  classifyButton: byId("classify-button"),
  clearButton: byId("clear-button"),
  examples: byId("examples-list"),
  result: byId("result"),
  resultEmpty: byId("result-empty"),
  resultLoading: byId("result-loading"),
  resultBody: byId("result-body"),
  resultError: byId("result-error"),
  topIntent: byId("top-intent"),
  scenarioBadge: byId("scenario-badge"),
  confidenceValue: byId("confidence-value"),
  confidenceMeter: byId("confidence-meter"),
  confidenceFill: byId("confidence-fill"),
  topK: byId("top-k"),
  reviewBanner: byId("review-banner"),
  latencyChip: byId("latency-chip"),
  compareControl: byId("compare-control"),
  compareToggle: byId("compare-toggle"),
  compare: byId("compare"),
  compareAgreement: byId("compare-agreement"),
  compareDefaultIntent: byId("compare-default-intent"),
  compareDefaultScenario: byId("compare-default-scenario"),
  compareDefaultConfidence: byId("compare-default-confidence"),
  compareBaselineIntent: byId("compare-baseline-intent"),
  compareBaselineScenario: byId("compare-baseline-scenario"),
  compareBaselineConfidence: byId("compare-baseline-confidence"),
  errorDetail: byId("error-detail"),
  errorHint: byId("error-hint"),
  retryButton: byId("retry-button"),
  modelId: byId("model-id"),
  modelVersion: byId("model-version"),
  modelQuantization: byId("model-quantization"),
  modelParameters: byId("model-parameters"),
  modelLicense: byId("model-license"),
  modelTemperature: byId("model-temperature"),
  modelThreshold: byId("model-threshold"),
  modelEce: byId("model-ece"),
  modelBaseline: byId("model-baseline"),
};

class ApiError extends Error {
  constructor(status, title, detail) {
    super(detail || title);
    this.status = status;
    this.title = title;
  }
}

function safeGet(key) {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeSet(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* Private mode or blocked storage: keep the in-memory preference. */
  }
}

function defaultLanguage() {
  const stored = safeGet(LANG_KEY);
  if (stored === "es" || stored === "en") {
    return stored;
  }
  const browser = (navigator.language || "").toLowerCase();
  if (browser.startsWith("es")) {
    return "es";
  }
  if (browser.startsWith("en")) {
    return "en";
  }
  return "es";
}

function defaultTheme() {
  const stored = safeGet(THEME_KEY);
  if (stored === "light" || stored === "dark" || stored === "system") {
    return stored;
  }
  return "system";
}

const state = {
  lang: defaultLanguage(),
  theme: defaultTheme(),
  model: null,
  busy: false,
  lastText: "",
  compare: false,
};

function t(key, variables = {}) {
  const catalogue = STRINGS[state.lang] || STRINGS.es;
  const parts = key.split(".");
  let value = catalogue;
  for (const part of parts) {
    if (value && typeof value === "object" && part in value) {
      value = value[part];
    } else {
      value = undefined;
      break;
    }
  }
  if (typeof value !== "string") {
    return key;
  }
  return value.replace(/\{(\w+)\}/g, (match, name) =>
    name in variables ? String(variables[name]) : match,
  );
}

function scenarioLabel(scenario) {
  const catalogue = STRINGS[state.lang] || STRINGS.es;
  return (catalogue.scenarios && catalogue.scenarios[scenario]) || scenario || "—";
}

function formatPercent(value) {
  const safe = Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
  return `${Math.round(safe * 100)}%`;
}

function formatNumber(value, digits = 0) {
  if (!Number.isFinite(value)) {
    return "—";
  }
  return new Intl.NumberFormat(state.lang === "es" ? "es-ES" : "en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

function applyStaticTranslations() {
  for (const node of document.querySelectorAll("[data-i18n]")) {
    const key = node.getAttribute("data-i18n");
    if (key) {
      node.textContent = t(key);
    }
  }
  for (const node of document.querySelectorAll("[data-i18n-placeholder]")) {
    const key = node.getAttribute("data-i18n-placeholder");
    if (key) {
      node.setAttribute("placeholder", t(key));
    }
  }
  for (const node of document.querySelectorAll("[data-i18n-aria-label]")) {
    const key = node.getAttribute("data-i18n-aria-label");
    if (key) {
      node.setAttribute("aria-label", t(key));
    }
  }
  elements.html.setAttribute("lang", state.lang);
  elements.langToggle.textContent = state.lang === "es" ? "EN" : "ES";
  document.title = `stc · ${t("app.title")}`;
  updateCounter();
  updateButtonState();
}

function applyTheme() {
  elements.html.setAttribute("data-theme", state.theme);
  safeSet(THEME_KEY, state.theme);
}

function resolvedTheme() {
  if (state.theme === "system") {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  return state.theme;
}

function toggleTheme() {
  state.theme = resolvedTheme() === "dark" ? "light" : "dark";
  applyTheme();
}

function toggleLanguage() {
  state.lang = state.lang === "es" ? "en" : "es";
  safeSet(LANG_KEY, state.lang);
  applyStaticTranslations();
  renderModelPanel();
  if (state.lastText) {
    const primary = currentPrimaryResult();
    if (primary) {
      renderResult(primary);
    }
    const baseline = currentBaselineResult();
    if (baseline && state.compare) {
      renderCompare(primary, baseline);
    }
  }
}

function currentPrimaryResult() {
  return state.primary || null;
}

function currentBaselineResult() {
  return state.baseline || null;
}

function updateCounter() {
  if (!elements.input || !elements.counter) {
    return;
  }
  elements.counter.textContent = t("input.counter", {
    count: elements.input.value.length,
    max: MAX_CHARS,
  });
}

function updateButtonState() {
  if (!elements.classifyButton) {
    return;
  }
  elements.classifyButton.disabled =
    state.busy || elements.input.value.trim().length === 0;
  elements.classifyButton.setAttribute("aria-disabled", String(elements.classifyButton.disabled));
}

function setResultState(name) {
  elements.result.dataset.state = name;
  elements.result.setAttribute("aria-busy", name === "loading" ? "true" : "false");
  elements.resultEmpty.hidden = name !== "empty";
  elements.resultLoading.hidden = name !== "loading";
  elements.resultBody.hidden = name !== "ready";
  elements.resultError.hidden = name !== "error";
}

function resetResult() {
  state.lastText = "";
  state.primary = null;
  state.baseline = null;
  elements.compare.hidden = true;
  elements.compareToggle.checked = false;
  state.compare = false;
  setResultState("empty");
}

function renderModelPanel() {
  const model = state.model;
  const unavailable = t("modelPanel.unavailable");
  if (!model) {
    elements.modelChip.textContent = t("header.modelLoading");
    return;
  }
  elements.modelChip.textContent = `${model.id} · ${model.version}`;
  elements.modelId.textContent = model.id || unavailable;
  elements.modelVersion.textContent = model.version || unavailable;
  elements.modelQuantization.textContent = model.quantization || "fp32";
  elements.modelParameters.textContent = Number.isFinite(model.parameters)
    ? formatNumber(model.parameters)
    : unavailable;
  elements.modelLicense.textContent = model.license || unavailable;
  elements.modelTemperature.textContent = Number.isFinite(model.temperature)
    ? formatNumber(model.temperature, 2)
    : unavailable;
  elements.modelThreshold.textContent = Number.isFinite(model.review_threshold)
    ? formatNumber(model.review_threshold, 2)
    : unavailable;
  if (Number.isFinite(model.dev_ece_before) && Number.isFinite(model.dev_ece_after)) {
    elements.modelEce.textContent = `${formatNumber(model.dev_ece_before, 3)} → ${formatNumber(
      model.dev_ece_after,
      3,
    )}`;
  } else {
    elements.modelEce.textContent = unavailable;
  }
  elements.modelBaseline.textContent = model.baseline_available
    ? t("modelPanel.yes")
    : t("modelPanel.no");
  elements.compareControl.hidden = !model.baseline_available;
}

async function loadModelInfo() {
  try {
    const response = await fetch("/v1/model", { headers: { accept: "application/json" } });
    if (!response.ok) {
      throw new Error(`model info returned ${response.status}`);
    }
    state.model = await response.json();
    renderModelPanel();
  } catch {
    elements.modelChip.textContent = t("header.modelLoading");
  }
}

function candidateRow(label) {
  const item = document.createElement("li");
  item.className = "candidate";

  const main = document.createElement("div");
  main.className = "candidate-main";
  const intent = document.createElement("code");
  intent.className = "candidate-intent";
  intent.textContent = label.intent;
  const scenario = document.createElement("span");
  scenario.className = "candidate-scenario";
  scenario.textContent = scenarioLabel(label.scenario);
  main.append(intent, scenario);

  const score = document.createElement("span");
  score.className = "candidate-score";
  score.textContent = formatPercent(label.probability);

  const bar = document.createElement("div");
  bar.className = "candidate-bar";
  const fill = document.createElement("span");
  fill.style.width = formatPercent(label.probability);
  bar.append(fill);

  item.append(main, score, bar);
  return item;
}

function renderResult(payload) {
  const prediction = payload.predictions && payload.predictions[0];
  if (!prediction) {
    showError(new Error(t("error.generic")));
    return;
  }
  state.primary = payload;
  elements.topIntent.textContent = prediction.intent;
  elements.scenarioBadge.textContent = scenarioLabel(prediction.scenario);
  elements.confidenceValue.textContent = formatPercent(prediction.confidence);

  const percent = Math.round(Math.min(1, Math.max(0, prediction.confidence || 0)) * 100);
  elements.confidenceMeter.setAttribute("aria-valuenow", String(prediction.confidence || 0));
  elements.confidenceMeter.setAttribute("aria-valuetext", `${percent}%`);
  elements.confidenceFill.style.width = `${percent}%`;

  elements.topK.replaceChildren();
  for (const label of prediction.top_k || []) {
    elements.topK.append(candidateRow(label));
  }

  elements.reviewBanner.hidden = !prediction.needs_review;
  elements.latencyChip.textContent = t("result.latency", {
    ms: formatNumber(payload.latency_ms || 0, 1),
  });
  elements.compareControl.hidden = !(state.model && state.model.baseline_available);
  elements.compareToggle.checked = state.compare;
  setResultState("ready");
}

function renderCompare(primaryPayload, baselinePayload) {
  const primary = primaryPayload && primaryPayload.predictions && primaryPayload.predictions[0];
  const baseline = baselinePayload && baselinePayload.predictions && baselinePayload.predictions[0];
  if (!primary || !baseline) {
    elements.compare.hidden = true;
    return;
  }
  elements.compareDefaultIntent.textContent = primary.intent;
  elements.compareDefaultScenario.textContent = scenarioLabel(primary.scenario);
  elements.compareDefaultConfidence.textContent = t("compare.confidence", {
    value: formatPercent(primary.confidence),
  });
  elements.compareBaselineIntent.textContent = baseline.intent;
  elements.compareBaselineScenario.textContent = scenarioLabel(baseline.scenario);
  elements.compareBaselineConfidence.textContent = t("compare.confidence", {
    value: formatPercent(baseline.confidence),
  });
  const agree = primary.intent === baseline.intent;
  elements.compareAgreement.textContent = t(agree ? "compare.agree" : "compare.disagree");
  elements.compareAgreement.dataset.state = agree ? "agree" : "disagree";
  elements.compare.hidden = false;
}

function showError(error) {
  let detail = t("error.generic");
  let hint = "";
  if (error instanceof ApiError) {
    if (error.status === 503 && (error.title || "").toLowerCase().includes("model")) {
      detail = t("error.model");
      hint = "stc model pull";
    } else if (error.status === 503) {
      detail = t("error.service");
    } else {
      detail = error.message || detail;
    }
  } else if (error instanceof TypeError) {
    detail = t("error.unreachable");
    hint = "stc serve";
  } else if (error instanceof Error && error.message) {
    detail = error.message;
  }
  elements.errorDetail.textContent = detail;
  elements.errorHint.textContent = hint;
  elements.errorHint.hidden = hint.length === 0;
  elements.compare.hidden = true;
  setResultState("error");
}

async function predict(text, model) {
  let response;
  try {
    response = await fetch("/v1/predict", {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/json" },
      body: JSON.stringify({ texts: [text], top_k: 3, model }),
    });
  } catch {
    throw new TypeError("network request failed");
  }
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    throw new ApiError(
      response.status,
      (payload && payload.title) || "",
      (payload && payload.detail) || "",
    );
  }
  return payload;
}

async function classify() {
  const text = elements.input.value.trim();
  if (!text || state.busy) {
    return;
  }
  state.busy = true;
  state.lastText = text;
  updateButtonState();
  setResultState("loading");

  const wantsCompare = Boolean(state.compare && state.model && state.model.baseline_available);
  try {
    const primary = await predict(text, "default");
    let baseline = null;
    if (wantsCompare) {
      baseline = await predict(text, "baseline");
    }
    state.baseline = baseline;
    renderResult(primary);
    if (baseline) {
      renderCompare(primary, baseline);
    } else {
      elements.compare.hidden = true;
    }
  } catch (error) {
    state.primary = null;
    state.baseline = null;
    showError(error);
  } finally {
    state.busy = false;
    updateButtonState();
  }
}

function updateOnlineStatus() {
  elements.offlineBanner.hidden = navigator.onLine;
}

function bindEvents() {
  elements.form.addEventListener("submit", (event) => {
    event.preventDefault();
    classify();
  });
  elements.input.addEventListener("input", () => {
    updateCounter();
    updateButtonState();
  });
  elements.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      classify();
    }
  });
  elements.clearButton.addEventListener("click", () => {
    elements.input.value = "";
    updateCounter();
    updateButtonState();
    resetResult();
    elements.input.focus();
  });
  elements.examples.addEventListener("click", (event) => {
    const button = event.target.closest(".example-chip");
    if (!button) {
      return;
    }
    elements.input.value = button.dataset.example || button.textContent.trim();
    updateCounter();
    updateButtonState();
    elements.input.focus();
    elements.input.setSelectionRange(elements.input.value.length, elements.input.value.length);
  });
  elements.themeToggle.addEventListener("click", toggleTheme);
  elements.langToggle.addEventListener("click", toggleLanguage);
  elements.compareToggle.addEventListener("change", () => {
    state.compare = elements.compareToggle.checked;
    if (state.compare && !state.busy && state.lastText) {
      classify();
    } else if (!state.compare) {
      elements.compare.hidden = true;
    }
  });
  elements.retryButton.addEventListener("click", () => {
    if (state.lastText) {
      elements.input.value = state.lastText;
      updateCounter();
      classify();
    }
  });
  window.addEventListener("online", updateOnlineStatus);
  window.addEventListener("offline", updateOnlineStatus);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (state.theme === "system") {
      applyTheme();
    }
  });
}

applyTheme();
applyStaticTranslations();
bindEvents();
updateOnlineStatus();
resetResult();
loadModelInfo();
