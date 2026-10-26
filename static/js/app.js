// FloraVision AI - Core Frontend Logic

let currentImageFile = null;
let currentImageDataUrl = null;
let currentInferenceResult = null;
let currentOverlayMode = "annotated"; // "annotated" or "original"
let speciesDatabase = {};
let allSamples = [];
let chartAccuracy = null;
let chartLoss = null;
let webcamStream = null;

document.addEventListener("DOMContentLoaded", () => {
  initDropArea();
  initPasteHandler();
  fetchHealth();
  fetchSamples();
  fetchSpeciesCatalog();
  fetchHistory();
  initMetrics();
});

// ==========================================
// TAB NAVIGATION
// ==========================================
function switchTab(tabId) {
  document.querySelectorAll(".tab-content").forEach(el => {
    el.classList.add("hidden");
    el.classList.remove("block");
  });

  const target = document.getElementById(`tab-${tabId}`);
  if (target) {
    target.classList.remove("hidden");
    target.classList.add("block");
  }

  document.querySelectorAll(".nav-tab").forEach(btn => {
    btn.classList.remove("active-tab", "text-emerald-400");
    btn.classList.add("text-slate-400");
  });

  const activeBtn = document.getElementById(`tab-btn-${tabId}`);
  if (activeBtn) {
    activeBtn.classList.add("active-tab");
    activeBtn.classList.remove("text-slate-400");
  }

  // Trigger chart re-render if switching to metrics tab
  if (tabId === "metrics") {
    setTimeout(() => {
      if (chartAccuracy) chartAccuracy.resize();
      if (chartLoss) chartLoss.resize();
    }, 150);
  }
}

// ==========================================
// SYSTEM HEALTH CHECK
// ==========================================
async function fetchHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const badge = document.getElementById("system-status-badge");
    const text = document.getElementById("system-status-text");

    if (data.status === "healthy") {
      badge.className = "flex items-center space-x-2 px-3 py-1.5 rounded-full bg-emerald-950/60 border border-emerald-800 text-xs text-emerald-300";
      text.textContent = `YOLO Online (${data.total_species_supported} Species)`;
    }
  } catch (err) {
    console.warn("Health check deferred:", err);
  }
}

// ==========================================
// SAMPLE CARDS (1-Click Test Drive)
// ==========================================
async function fetchSamples() {
  try {
    const res = await fetch("/api/samples");
    const data = await res.json();
    allSamples = data.samples || [];

    const grid = document.getElementById("sample-cards-grid");
    if (!grid) return;
    grid.innerHTML = "";

    allSamples.forEach(item => {
      const card = document.createElement("div");
      card.className = "sample-card bg-slate-950/80 border border-slate-800/80 hover:border-emerald-500/50 rounded-xl p-2 cursor-pointer transition text-center group";
      card.onclick = () => loadSample(item.id);

      card.innerHTML = `
        <div class="w-full aspect-square rounded-lg overflow-hidden bg-slate-900 mb-1.5 relative">
          <img src="${item.url}" alt="${item.name}" class="w-full h-full object-cover group-hover:scale-110 transition duration-300">
        </div>
        <p class="text-[11px] font-bold text-slate-200 truncate group-hover:text-emerald-400 transition">${item.name}</p>
      `;
      grid.appendChild(card);
    });
  } catch (e) {
    console.error("Error fetching samples:", e);
  }
}

async function loadSample(speciesId) {
  switchTab("classifier");
  showLoading();

  try {
    const sampleUrl = `/static/samples/${speciesId}.jpg`;
    // Fetch image as blob
    const res = await fetch(sampleUrl);
    if (!res.ok) throw new Error("Sample file not found");
    const blob = await res.blob();
    const file = new File([blob], `${speciesId}.jpg`, { type: "image/jpeg" });

    // Set preview
    const reader = new FileReader();
    reader.onload = (e) => {
      currentImageDataUrl = e.target.result;
      currentImageFile = file;
      showInputPreview(file.name, currentImageDataUrl);
      // Run classification
      triggerClassification();
    };
    reader.readAsDataURL(file);
  } catch (err) {
    console.error("Error loading sample:", err);
    alert("Could not load sample image: " + err.message);
    hideLoading();
  }
}

// ==========================================
// FILE INPUT & DRAG/DROP
// ==========================================
function initDropArea() {
  const dropArea = document.getElementById("drop-area");
  const fileInput = document.getElementById("file-input");

  if (!dropArea || !fileInput) return;

  dropArea.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  ["dragenter", "dragover"].forEach(eventName => {
    dropArea.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropArea.classList.add("border-emerald-400", "bg-emerald-500/10");
    });
  });

  ["dragleave", "drop"].forEach(eventName => {
    dropArea.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropArea.classList.remove("border-emerald-400", "bg-emerald-500/10");
    });
  });

  dropArea.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });
}

function initPasteHandler() {
  window.addEventListener("paste", (e) => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (const item of items) {
      if (item.type.indexOf("image") !== -1) {
        const file = item.getAsFile();
        handleSelectedFile(file);
        break;
      }
    }
  });
}

function handleSelectedFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please select a valid image file (JPG, PNG, WEBP).");
    return;
  }

  currentImageFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    currentImageDataUrl = e.target.result;
    showInputPreview(file.name, currentImageDataUrl);
  };
  reader.readAsDataURL(file);
}

function showInputPreview(filename, dataUrl) {
  const box = document.getElementById("input-preview-box");
  const thumb = document.getElementById("input-thumbnail");
  const nameEl = document.getElementById("input-filename");

  thumb.src = dataUrl;
  nameEl.textContent = filename;
  box.classList.remove("hidden");
}

// ==========================================
// WEBCAM INTEGRATION
// ==========================================
function setInputMode(mode) {
  const btnUpload = document.getElementById("btn-mode-upload");
  const btnCamera = document.getElementById("btn-mode-camera");
  const dropArea = document.getElementById("drop-area");
  const cameraArea = document.getElementById("camera-area");

  if (mode === "camera") {
    btnCamera.className = "px-3 py-1 rounded-lg text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30";
    btnUpload.className = "px-3 py-1 rounded-lg text-xs font-semibold bg-slate-800 text-slate-400 hover:text-white";
    dropArea.classList.add("hidden");
    cameraArea.classList.remove("hidden");
    startWebcam();
  } else {
    btnUpload.className = "px-3 py-1 rounded-lg text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30";
    btnCamera.className = "px-3 py-1 rounded-lg text-xs font-semibold bg-slate-800 text-slate-400 hover:text-white";
    dropArea.classList.remove("hidden");
    cameraArea.classList.add("hidden");
    stopWebcam();
  }
}

async function startWebcam() {
  try {
    const video = document.getElementById("webcam-video");
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    video.srcObject = webcamStream;
  } catch (err) {
    alert("Could not access webcam: " + err.message);
    setInputMode("upload");
  }
}

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach(track => track.stop());
    webcamStream = null;
  }
}

function captureWebcam() {
  const video = document.getElementById("webcam-video");
  if (!video) return;

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
  currentImageDataUrl = dataUrl;
  currentImageFile = null;

  showInputPreview("webcam_snapshot.jpg", dataUrl);
  setInputMode("upload");
  triggerClassification();
}

// ==========================================
// INFERENCE & CLASSIFICATION
// ==========================================
async function triggerClassification() {
  if (!currentImageDataUrl && !currentImageFile) {
    alert("Please upload an image first.");
    return;
  }

  showLoading();

  try {
    let res;
    if (currentImageFile) {
      const formData = new FormData();
      formData.append("image", currentImageFile);
      res = await fetch("/api/classify", {
        method: "POST",
        body: formData
      });
    } else {
      res = await fetch("/api/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: currentImageDataUrl, filename: "webcam.jpg" })
      });
    }

    const data = await res.json();
    if (!res.ok || data.error) {
      throw new Error(data.error || "Inference error");
    }

    currentInferenceResult = data;
    renderResults(data);
    fetchHistory(); // Refresh history log

    // Trigger celebration confetti on high confidence!
    if (data.confidence >= 95 && typeof confetti === "function") {
      confetti({
        particleCount: 50,
        spread: 60,
        origin: { y: 0.7 }
      });
    }

  } catch (err) {
    console.error("Classification failed:", err);
    alert("Classification failed: " + err.message);
  } finally {
    hideLoading();
  }
}

function renderResults(data) {
  document.getElementById("results-empty-state").classList.add("hidden");
  document.getElementById("results-loading-state").classList.add("hidden");
  document.getElementById("results-active-card").classList.remove("hidden");

  // Hero Fields
  document.getElementById("res-common-name").textContent = data.common_name;
  document.getElementById("res-scientific-name").textContent = data.scientific_name;
  document.getElementById("res-confidence-pct").textContent = `${data.confidence}%`;
  document.getElementById("res-inference-time").textContent = `${data.inference_time_ms}ms`;

  const bot = data.botanical || {};
  document.getElementById("res-family-order").textContent = `Family: ${bot.family || data.family} | Order: ${bot.order || 'Plantae'}`;
  document.getElementById("res-care-sunlight").textContent = bot.sunlight || "Full Sun";
  document.getElementById("res-care-watering").textContent = bot.watering || "Moderate";
  document.getElementById("res-care-temp").textContent = bot.temperature || "15°C - 28°C";
  document.getElementById("res-care-season").textContent = bot.blooming_season || "Spring - Autumn";

  document.getElementById("res-symbolism").textContent = bot.symbolism || "Beauty and resilience.";
  document.getElementById("res-medicinal").textContent = bot.medicinal_uses || "Rich in botanical bioflavonoids.";
  document.getElementById("res-fun-fact").textContent = bot.fun_fact || "A marvel of floral geometry.";

  // Pet Toxicity Badge
  const toxEl = document.getElementById("res-pet-toxicity");
  const toxText = bot.toxicity_pets || "Non-toxic";
  if (toxText.toLowerCase().includes("toxic to cats") || toxText.toLowerCase().includes("toxic")) {
    toxEl.className = "px-3 py-1 rounded-full text-xs font-bold bg-rose-500/20 text-rose-300 border border-rose-500/30";
    toxEl.innerHTML = `<i class="fa-solid fa-triangle-exclamation mr-1"></i> ${toxText}`;
  } else {
    toxEl.className = "px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
    toxEl.innerHTML = `🐾 Pet Friendly`;
  }

  // Display Image with YOLO box
  setImageOverlayMode("annotated");

  // Top-5 Breakdown Bars
  const top5Container = document.getElementById("res-top5-container");
  top5Container.innerHTML = "";
  (data.top_5 || []).forEach((item, idx) => {
    const isTop = idx === 0;
    const bar = document.createElement("div");
    bar.className = "space-y-1";
    bar.innerHTML = `
      <div class="flex items-center justify-between text-xs">
        <span class="font-medium ${isTop ? 'text-emerald-300 font-bold' : 'text-slate-300'}">${item.common_name} <span class="italic text-slate-500 text-[11px]">(${item.scientific_name})</span></span>
        <span class="font-mono font-bold ${isTop ? 'text-emerald-400' : 'text-slate-400'}">${item.confidence}%</span>
      </div>
      <div class="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
        <div class="h-full rounded-full ${isTop ? 'bg-gradient-to-r from-emerald-500 to-teal-400' : 'bg-slate-600'} transition-all duration-700" style="width: ${item.confidence}%"></div>
      </div>
    `;
    top5Container.appendChild(bar);
  });

  // Color Palette Swatches
  const paletteContainer = document.getElementById("res-color-palette");
  paletteContainer.innerHTML = "";
  (data.palette || []).forEach(c => {
    const swatch = document.createElement("div");
    swatch.className = "flex-1 group relative";
    swatch.innerHTML = `
      <div class="w-full h-8 rounded-lg border border-slate-700 shadow" style="background-color: ${c.hex}"></div>
      <div class="text-center mt-1 text-[10px] font-mono text-slate-400">${c.percentage}%</div>
      <div class="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block px-2 py-1 bg-slate-950 text-[10px] text-white rounded border border-slate-700 whitespace-nowrap z-10">
        ${c.name} (${c.hex})
      </div>
    `;
    paletteContainer.appendChild(swatch);
  });

  // Morphology Geometry
  if (data.morphology) {
    document.getElementById("res-morph-circularity").textContent = data.morphology.circularity || "0.85";
    document.getElementById("res-morph-ratio").textContent = data.morphology.aspect_ratio || "1.00";
    document.getElementById("res-morph-petals").textContent = data.morphology.estimated_petals || "Radial";
  }
}

function setImageOverlayMode(mode) {
  currentOverlayMode = mode;
  const imgEl = document.getElementById("res-image-display");
  const btnAnnotated = document.getElementById("btn-view-annotated");
  const btnOriginal = document.getElementById("btn-view-original");

  if (!currentInferenceResult) return;

  if (mode === "annotated") {
    imgEl.src = currentInferenceResult.annotated_image;
    btnAnnotated.className = "px-2.5 py-1 rounded bg-emerald-500/20 text-emerald-300 font-semibold text-[11px]";
    btnOriginal.className = "px-2.5 py-1 rounded text-slate-400 hover:text-white font-semibold text-[11px]";
  } else {
    imgEl.src = currentImageDataUrl;
    btnOriginal.className = "px-2.5 py-1 rounded bg-emerald-500/20 text-emerald-300 font-semibold text-[11px]";
    btnAnnotated.className = "px-2.5 py-1 rounded text-slate-400 hover:text-white font-semibold text-[11px]";
  }
}

function showLoading() {
  document.getElementById("results-empty-state").classList.add("hidden");
  document.getElementById("results-active-card").classList.add("hidden");
  document.getElementById("results-loading-state").classList.remove("hidden");
}

function hideLoading() {
  document.getElementById("results-loading-state").classList.add("hidden");
}

function resetClassifier() {
  currentImageFile = null;
  currentImageDataUrl = null;
  currentInferenceResult = null;
  document.getElementById("input-preview-box").classList.add("hidden");
  document.getElementById("results-active-card").classList.add("hidden");
  document.getElementById("results-loading-state").classList.add("hidden");
  document.getElementById("results-empty-state").classList.remove("hidden");
}

// ==========================================
// VOICE SYNTHESIS (TTS)
// ==========================================
function speakFlowerName() {
  if (!currentInferenceResult) return;
  if (!("speechSynthesis" in window)) {
    alert("Speech synthesis not supported in this browser.");
    return;
  }
  const text = `${currentInferenceResult.common_name}, scientifically known as ${currentInferenceResult.scientific_name}. Family ${currentInferenceResult.family}.`;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 0.95;
  utterance.pitch = 1.0;
  window.speechSynthesis.speak(utterance);
}

// ==========================================
// PRINT / PDF REPORT CERTIFICATE
// ==========================================
function printPlantReport() {
  if (!currentInferenceResult) return;
  const cert = document.getElementById("print-certificate");
  const res = currentInferenceResult;
  const bot = res.botanical || {};

  cert.innerHTML = `
    <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; border: 4px double #059669; padding: 32px; border-radius: 12px;">
      <div style="text-align: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 16px; margin-bottom: 24px;">
        <h1 style="color: #065f46; margin: 0; font-size: 26px;">FloraVision AI • Botanical Identification Certificate</h1>
        <p style="color: #64748b; font-size: 13px; margin: 4px 0 0 0;">YOLOv8 Deep Learning Species Recognition System</p>
      </div>

      <div style="display: flex; gap: 24px; align-items: center; margin-bottom: 24px;">
        <img src="${res.annotated_image}" style="width: 260px; height: 260px; object-fit: contain; border: 1px solid #cbd5e1; border-radius: 8px;">
        <div style="flex: 1;">
          <h2 style="font-size: 28px; margin: 0; color: #0f172a;">${res.common_name}</h2>
          <p style="font-style: italic; color: #047857; font-size: 18px; margin: 4px 0 12px 0;">${res.scientific_name}</p>
          <p><strong>Taxonomic Family:</strong> ${bot.family || res.family}</p>
          <p><strong>Identification Confidence:</strong> <span style="color: #059669; font-weight: bold; font-size: 18px;">${res.confidence}%</span></p>
          <p><strong>Identified On:</strong> ${new Date().toLocaleString()}</p>
        </div>
      </div>

      <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 16px; border-radius: 8px; margin-bottom: 20px;">
        <h3 style="margin-top: 0; color: #1e293b; font-size: 16px;">Cultivation & Care Parameters</h3>
        <p><strong>Sunlight Requirement:</strong> ${bot.sunlight || 'Full Sun'}</p>
        <p><strong>Watering Regimen:</strong> ${bot.watering || 'Moderate'}</p>
        <p><strong>Optimal Soil:</strong> ${bot.soil_type || 'Loamy, well-draining'}</p>
        <p><strong>Blooming Season:</strong> ${bot.blooming_season || 'Spring-Autumn'}</p>
        <p><strong>Pet Toxicity:</strong> ${bot.toxicity_pets || 'Non-toxic'}</p>
      </div>

      <div style="border-top: 1px solid #cbd5e1; padding-top: 12px; font-size: 11px; color: #94a3b8; text-align: center;">
        FloraVision AI Vision Engine • Validated Botanical Classification
      </div>
    </div>
  `;

  window.print();
}

// ==========================================
// SPECIES ENCYCLOPEDIA CATALOG
// ==========================================
async function fetchSpeciesCatalog() {
  try {
    const res = await fetch("/api/species");
    const data = await res.json();
    const species = data.species || [];

    // Populate comparison dropdowns as well
    const s1 = document.getElementById("compare-select-1");
    const s2 = document.getElementById("compare-select-2");
    if (s1 && s2) {
      s1.innerHTML = "";
      s2.innerHTML = "";
      species.forEach((sp, idx) => {
        const opt1 = new Option(sp.common_name, sp.id);
        const opt2 = new Option(sp.common_name, sp.id);
        if (idx === 0) opt1.selected = true;
        if (idx === 1) opt2.selected = true;
        s1.add(opt1);
        s2.add(opt2);
      });
      loadComparison();
    }

    renderCatalogGrid(species);
  } catch (e) {
    console.error("Error loading catalog:", e);
  }
}

function renderCatalogGrid(speciesList) {
  const grid = document.getElementById("catalog-grid");
  if (!grid) return;
  grid.innerHTML = "";

  speciesList.forEach(sp => {
    const card = document.createElement("div");
    card.className = "bg-slate-900 border border-slate-800 rounded-3xl p-5 hover:border-emerald-500/50 transition duration-300 flex flex-col justify-between group";

    card.innerHTML = `
      <div>
        <div class="w-full aspect-square rounded-2xl overflow-hidden bg-slate-950 mb-4 border border-slate-800 relative">
          <img src="${sp.sample_image}" alt="${sp.common_name}" class="w-full h-full object-cover group-hover:scale-105 transition duration-300">
          <span class="absolute top-2.5 right-2.5 px-2 py-0.5 rounded-full text-[10px] font-bold bg-slate-950/80 backdrop-blur-md text-emerald-400 border border-emerald-500/30">
            ${sp.family}
          </span>
        </div>
        <h3 class="font-display font-bold text-lg text-white group-hover:text-emerald-400 transition">${sp.common_name}</h3>
        <p class="font-serif italic text-xs text-emerald-400/90 mb-3">${sp.scientific_name}</p>
        <div class="space-y-1.5 text-xs text-slate-400">
          <p><strong class="text-slate-300">Season:</strong> ${sp.blooming_season}</p>
          <p><strong class="text-slate-300">Sun:</strong> ${sp.sunlight}</p>
        </div>
      </div>
      <div class="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between">
        <button onclick="loadSample('${sp.id}')" class="text-xs font-semibold text-emerald-400 hover:text-emerald-300 flex items-center space-x-1">
          <span>Test in Classifier</span>
          <i class="fa-solid fa-arrow-right text-[10px]"></i>
        </button>
      </div>
    `;
    grid.appendChild(card);
  });
}

function filterCatalog() {
  const query = document.getElementById("catalog-search").value.toLowerCase();
  const cards = document.querySelectorAll("#catalog-grid > div");

  cards.forEach(card => {
    const text = card.textContent.toLowerCase();
    if (text.includes(query)) {
      card.style.display = "flex";
    } else {
      card.style.display = "none";
    }
  });
}

// ==========================================
// SPECIES COMPARATOR
// ==========================================
async function loadComparison() {
  const f1 = document.getElementById("compare-select-1").value;
  const f2 = document.getElementById("compare-select-2").value;

  try {
    const res = await fetch(`/api/compare?flower1=${f1}&flower2=${f2}`);
    const data = await res.json();
    renderComparisonCards(data.flower1, data.flower2);
  } catch (err) {
    console.error("Comparison load error:", err);
  }
}

function renderComparisonCards(fl1, fl2) {
  const container = document.getElementById("comparison-view");
  if (!container) return;

  const buildCard = (fl) => `
    <div class="bg-slate-900 border border-slate-800 rounded-3xl p-6 space-y-4">
      <div class="flex items-center space-x-4">
        <img src="/static/samples/${fl.id}.jpg" class="w-20 h-20 rounded-2xl object-cover border border-slate-700">
        <div>
          <h3 class="font-display font-extrabold text-2xl text-white">${fl.common_name}</h3>
          <p class="font-serif italic text-emerald-400 text-sm">${fl.scientific_name}</p>
          <span class="text-xs text-slate-400 font-mono">Family: ${fl.family}</span>
        </div>
      </div>

      <div class="space-y-2 text-xs">
        <div class="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
          <strong class="text-slate-300 block mb-0.5">☀️ Sunlight Need:</strong>
          <span class="text-slate-400">${fl.sunlight}</span>
        </div>
        <div class="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
          <strong class="text-slate-300 block mb-0.5">💧 Watering Frequency:</strong>
          <span class="text-slate-400">${fl.watering}</span>
        </div>
        <div class="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
          <strong class="text-slate-300 block mb-0.5">🌱 Soil Preference:</strong>
          <span class="text-slate-400">${fl.soil_type}</span>
        </div>
        <div class="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
          <strong class="text-slate-300 block mb-0.5">🐾 Pet Toxicity:</strong>
          <span class="text-slate-400">${fl.toxicity_pets}</span>
        </div>
        <div class="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
          <strong class="text-slate-300 block mb-0.5">✨ Meaning & Symbolism:</strong>
          <span class="text-slate-400">${fl.symbolism}</span>
        </div>
      </div>
    </div>
  `;

  container.innerHTML = buildCard(fl1) + buildCard(fl2);
}

// ==========================================
// MODEL METRICS & CHARTS (VIVA READY)
// ==========================================
async function initMetrics() {
  try {
    const res = await fetch("/api/metrics");
    const data = await res.json();

    renderConfusionMatrix(data.species_labels, data.confusion_matrix);
    renderClassMetrics(data.class_metrics);
    renderCharts(data.epochs, data.train_accuracy, data.val_accuracy, data.loss);
  } catch (err) {
    console.error("Metrics load error:", err);
  }
}

function renderConfusionMatrix(labels, matrix) {
  const container = document.getElementById("confusion-matrix-table");
  if (!container) return;

  let html = `<table class="w-full text-center border-collapse"><thead><tr><th class="p-2 text-left text-slate-500">True \\ Pred</th>`;
  labels.forEach(lbl => {
    html += `<th class="p-2 text-slate-300 font-bold">${lbl.slice(0, 4)}</th>`;
  });
  html += `</tr></thead><tbody>`;

  matrix.forEach((row, i) => {
    html += `<tr class="border-t border-slate-800"><td class="p-2 text-left font-bold text-slate-300">${labels[i]}</td>`;
    row.forEach((val, j) => {
      const isDiag = i === j;
      const bg = isDiag ? 'bg-emerald-500/20 text-emerald-300 font-bold' : val > 0 ? 'text-slate-400' : 'text-slate-600';
      html += `<td class="p-2 ${bg}">${val}%</td>`;
    });
    html += `</tr>`;
  });

  html += `</tbody></table>`;
  container.innerHTML = html;
}

function renderClassMetrics(metrics) {
  const container = document.getElementById("class-metrics-list");
  if (!container) return;

  container.innerHTML = "";
  metrics.forEach(m => {
    const el = document.createElement("div");
    el.className = "p-3 rounded-2xl bg-slate-950/60 border border-slate-800 flex items-center justify-between";
    el.innerHTML = `
      <div>
        <p class="font-bold text-xs text-white">${m.species}</p>
        <span class="text-[10px] text-slate-500 font-mono">Support: ${m.support} samples</span>
      </div>
      <div class="flex items-center space-x-3 text-xs font-mono">
        <span class="text-slate-400">P: <strong class="text-white">${m.precision}%</strong></span>
        <span class="text-slate-400">R: <strong class="text-white">${m.recall}%</strong></span>
        <span class="text-emerald-400 font-bold">F1: ${m.f1_score}%</span>
      </div>
    `;
    container.appendChild(el);
  });
}

function renderCharts(epochs, trainAcc, valAcc, lossVals) {
  const ctxAcc = document.getElementById("chart-accuracy");
  const ctxLoss = document.getElementById("chart-loss");

  if (ctxAcc) {
    chartAccuracy = new Chart(ctxAcc, {
      type: "line",
      data: {
        labels: epochs,
        datasets: [
          {
            label: "Train Accuracy (%)",
            data: trainAcc,
            borderColor: "#34d399",
            backgroundColor: "rgba(52, 211, 153, 0.1)",
            tension: 0.3,
            fill: true
          },
          {
            label: "Validation Accuracy (%)",
            data: valAcc,
            borderColor: "#38bdf8",
            borderDash: [5, 5],
            tension: 0.3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#94a3b8", font: { size: 11 } } } },
        scales: {
          x: { ticks: { color: "#64748b", maxTicksLimit: 10 }, grid: { color: "#1e293b" } },
          y: { ticks: { color: "#64748b" }, grid: { color: "#1e293b" }, min: 65, max: 100 }
        }
      }
    });
  }

  if (ctxLoss) {
    chartLoss = new Chart(ctxLoss, {
      type: "line",
      data: {
        labels: epochs,
        datasets: [
          {
            label: "Cross-Entropy Loss",
            data: lossVals,
            borderColor: "#f43f5e",
            backgroundColor: "rgba(244, 63, 94, 0.1)",
            tension: 0.3,
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#94a3b8", font: { size: 11 } } } },
        scales: {
          x: { ticks: { color: "#64748b", maxTicksLimit: 10 }, grid: { color: "#1e293b" } },
          y: { ticks: { color: "#64748b" }, grid: { color: "#1e293b" }, min: 0 }
        }
      }
    });
  }
}

// ==========================================
// SESSION HISTORY
// ==========================================
async function fetchHistory() {
  try {
    const res = await fetch("/api/history");
    const data = await res.json();
    const list = document.getElementById("history-list");
    if (!list) return;

    if (!data.history || data.history.length === 0) {
      list.innerHTML = `
        <div class="text-center py-12 text-slate-500 text-sm">
          No classifications recorded yet. Run a classification to see your history!
        </div>
      `;
      return;
    }

    list.innerHTML = "";
    data.history.forEach(item => {
      const row = document.createElement("div");
      row.className = "bg-slate-900 border border-slate-800 rounded-2xl p-4 flex items-center justify-between";
      row.innerHTML = `
        <div class="flex items-center space-x-4">
          <img src="${item.thumbnail}" class="w-14 h-14 rounded-xl object-cover border border-slate-700">
          <div>
            <h4 class="font-display font-bold text-white text-sm">${item.common_name}</h4>
            <p class="font-serif italic text-xs text-emerald-400">${item.scientific_name}</p>
            <span class="text-[10px] text-slate-500 font-mono">${item.timestamp}</span>
          </div>
        </div>
        <div class="text-right">
          <span class="px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">
            ${item.confidence}% Match
          </span>
        </div>
      `;
      list.appendChild(row);
    });
  } catch (err) {
    console.error("Error fetching history:", err);
  }
}

async function clearHistory() {
  try {
    await fetch("/api/history/clear", { method: "POST" });
    fetchHistory();
  } catch (err) {
    console.error("Error clearing history:", err);
  }
}
