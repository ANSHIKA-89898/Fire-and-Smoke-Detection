/* app.js
 * Frontend logic for the Fire & Smoke Detection dashboard.
 * Talks to the Flask backend (server.py) via fetch() calls to /api/*.
 */

(() => {
  "use strict";

  // ------------------------------------------------------------------
  // Element references
  // ------------------------------------------------------------------
  const modeRadios = document.querySelectorAll('input[name="mode"]');
  const panels = {
    image: document.getElementById("panel-image"),
    video: document.getElementById("panel-video"),
    webcam: document.getElementById("panel-webcam"),
  };
  const frameSkipGroup = document.getElementById("frameSkipGroup");

  const confidenceSlider = document.getElementById("confidenceSlider");
  const confidenceValue = document.getElementById("confidenceValue");
  const iouSlider = document.getElementById("iouSlider");
  const iouValue = document.getElementById("iouValue");
  const alarmToggle = document.getElementById("alarmToggle");
  const cooldownSlider = document.getElementById("cooldownSlider");
  const cooldownValue = document.getElementById("cooldownValue");
  const frameSkipSlider = document.getElementById("frameSkipSlider");
  const frameSkipValue = document.getElementById("frameSkipValue");

  const modelBanner = document.getElementById("modelBanner");

  const statCurrentStatus = document.getElementById("statCurrentStatus");
  const statFireCount = document.getElementById("statFireCount");
  const statSmokeCount = document.getElementById("statSmokeCount");
  const statTotalAlerts = document.getElementById("statTotalAlerts");
  const statMaxConfidence = document.getElementById("statMaxConfidence");

  const imageInput = document.getElementById("imageInput");
  const imageResultArea = document.getElementById("imageResultArea");
  const imageResult = document.getElementById("imageResult");
  const imageAlertBanners = document.getElementById("imageAlertBanners");
  const imageDetectionList = document.getElementById("imageDetectionList");

  const videoInput = document.getElementById("videoInput");
  const startVideoBtn = document.getElementById("startVideoBtn");
  const videoProgress = document.getElementById("videoProgress");
  const videoProgressFill = document.getElementById("videoProgressFill");
  const videoLiveStats = document.getElementById("videoLiveStats");
  const videoResultArea = document.getElementById("videoResultArea");
  const videoResult = document.getElementById("videoResult");
  const videoSummary = document.getElementById("videoSummary");

  const startWebcamBtn = document.getElementById("startWebcamBtn");
  const stopWebcamBtn = document.getElementById("stopWebcamBtn");
  const webcamStatus = document.getElementById("webcamStatus");
  const webcamVideo = document.getElementById("webcamVideo");
  const webcamCanvas = document.getElementById("webcamCanvas");
  const webcamAnnotated = document.getElementById("webcamAnnotated");

  const currentAlert = document.getElementById("currentAlert");
  const latestDetection = document.getElementById("latestDetection");
  const latestTimestamp = document.getElementById("latestTimestamp");

  const historyEmpty = document.getElementById("historyEmpty");
  const historyTable = document.getElementById("historyTable");
  const historyTableBody = document.getElementById("historyTableBody");
  const clearHistoryBtn = document.getElementById("clearHistoryBtn");

  const alarmAudio = document.getElementById("alarmAudio");

  // ------------------------------------------------------------------
  // Shared dashboard state (derived from /api/history each refresh)
  // ------------------------------------------------------------------
  let webcamStream = null;
  let webcamLoopActive = false;
  let webcamTimer = null;

  function getThresholds() {
    return {
      confidence: parseFloat(confidenceSlider.value),
      iou: parseFloat(iouSlider.value),
    };
  }

  function playAlarmIfEnabled(newAlerts) {
    if (alarmToggle.checked && newAlerts && newAlerts.length > 0) {
      alarmAudio.currentTime = 0;
      alarmAudio.play().catch(() => {
        /* Autoplay can be blocked until the user interacts with the page;
           fail silently, visual alerts still work. */
      });
    }
  }

  // ------------------------------------------------------------------
  // Sidebar wiring
  // ------------------------------------------------------------------
  function switchMode(mode) {
    Object.entries(panels).forEach(([key, el]) => {
      el.classList.toggle("hidden", key !== mode);
    });
    frameSkipGroup.style.display = mode === "video" ? "block" : "none";
    if (mode !== "webcam" && webcamLoopActive) {
      stopWebcam();
    }
  }

  modeRadios.forEach((radio) => {
    radio.addEventListener("change", (e) => switchMode(e.target.value));
  });

  confidenceSlider.addEventListener("input", () => {
    confidenceValue.textContent = parseFloat(confidenceSlider.value).toFixed(2);
  });
  iouSlider.addEventListener("input", () => {
    iouValue.textContent = parseFloat(iouSlider.value).toFixed(2);
  });
  frameSkipSlider.addEventListener("input", () => {
    frameSkipValue.textContent = frameSkipSlider.value;
  });
  cooldownSlider.addEventListener("input", () => {
    cooldownValue.textContent = cooldownSlider.value;
    pushSettings();
  });
  alarmToggle.addEventListener("change", pushSettings);

  function pushSettings() {
    fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cooldown: parseFloat(cooldownSlider.value),
        alarm_enabled: alarmToggle.checked,
      }),
    }).catch(() => {});
  }

  // ------------------------------------------------------------------
  // Model status banner
  // ------------------------------------------------------------------
  async function checkModelStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      if (!data.model_ready) {
        modelBanner.classList.remove("hidden");
        modelBanner.innerHTML =
          `<strong>⚠️ Model not available.</strong><br>${escapeHtml(data.model_error || "")}` +
          `<br><br>Place your trained YOLO model at <code>${escapeHtml(data.model_path)}</code> and restart the server.`;
        [startVideoBtn, startWebcamBtn].forEach((btn) => (btn.disabled = true));
        imageInput.disabled = true;
      } else {
        modelBanner.classList.add("hidden");
      }
    } catch (err) {
      modelBanner.classList.remove("hidden");
      modelBanner.textContent = "Could not reach the backend server. Is server.py running?";
    }
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // ------------------------------------------------------------------
  // Image mode
  // ------------------------------------------------------------------
  imageInput.addEventListener("change", async () => {
    const file = imageInput.files[0];
    if (!file) return;

    const { confidence, iou } = getThresholds();
    const formData = new FormData();
    formData.append("image", file);
    formData.append("confidence", confidence);
    formData.append("iou", iou);

    imageAlertBanners.innerHTML = "";
    imageDetectionList.innerHTML = "";
    imageResultArea.classList.remove("hidden");
    imageResult.removeAttribute("src");

    try {
      const res = await fetch("/api/detect/image", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) {
        imageAlertBanners.innerHTML = `<div class="banner banner-error">${escapeHtml(data.error || "Detection failed.")}</div>`;
        return;
      }

      imageResult.src = "data:image/jpeg;base64," + data.annotated_image;

      const fireFound = data.detections.some((d) => d.class_name.toLowerCase() === "fire");
      const smokeFound = data.detections.some((d) => d.class_name.toLowerCase() === "smoke");

      let bannersHtml = "";
      if (fireFound) bannersHtml += `<div class="banner banner-error">🔥 FIRE DETECTED!</div>`;
      if (smokeFound) bannersHtml += `<div class="banner banner-warning">⚠️ SMOKE DETECTED!</div>`;
      if (!data.detections.length) bannersHtml = `<div class="banner banner-success">No fire or smoke detected in this image.</div>`;
      imageAlertBanners.innerHTML = bannersHtml;

      imageDetectionList.innerHTML = data.detections
        .map((d) => `<li>• <strong>${escapeHtml(d.class_name)}</strong> | ${d.confidence.toFixed(1)}%</li>`)
        .join("");

      statCurrentStatus.textContent = data.detections.length ? "Fire/Smoke Detected" : "Clear";

      playAlarmIfEnabled(data.new_alerts);
      await refreshHistoryAndStats();
    } catch (err) {
      imageAlertBanners.innerHTML = `<div class="banner banner-error">Request failed: ${escapeHtml(String(err))}</div>`;
    }
  });

  // ------------------------------------------------------------------
  // Video mode
  // ------------------------------------------------------------------
  startVideoBtn.addEventListener("click", async () => {
    const file = videoInput.files[0];
    if (!file) {
      alert("Please choose a video file first.");
      return;
    }

    const { confidence, iou } = getThresholds();
    const formData = new FormData();
    formData.append("video", file);
    formData.append("confidence", confidence);
    formData.append("iou", iou);
    formData.append("frame_skip", frameSkipSlider.value);

    startVideoBtn.disabled = true;
    videoProgress.classList.remove("hidden");
    videoProgressFill.style.width = "40%"; // indeterminate-ish feedback (single synchronous request)
    videoLiveStats.textContent = "Processing video on the server — this may take a while for longer clips...";
    videoResultArea.classList.add("hidden");

    try {
      const res = await fetch("/api/detect/video", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok) {
        videoLiveStats.textContent = data.error || "Video processing failed.";
        return;
      }

      videoProgressFill.style.width = "100%";
      const s = data.stats;
      videoLiveStats.textContent =
        `Frames processed: ${s.frames_processed} | Fire detections: ${s.fire_detections} | ` +
        `Smoke detections: ${s.smoke_detections} | Max confidence: ${s.max_confidence.toFixed(1)}%`;

      videoSummary.textContent =
        `Finished processing in ${s.duration_seconds.toFixed(1)}s. ` +
        (s.fire_detections === 0 && s.smoke_detections === 0
          ? "No fire or smoke detected."
          : "Fire/smoke was detected — check the annotated video and history below.");

      videoResult.src = data.annotated_video_url;
      videoResultArea.classList.remove("hidden");

      statCurrentStatus.textContent =
        s.fire_detections === 0 && s.smoke_detections === 0 ? "Clear" : "Fire/Smoke Detected";

      playAlarmIfEnabled(data.new_alerts);
      await refreshHistoryAndStats();
    } catch (err) {
      videoLiveStats.textContent = "Request failed: " + err;
    } finally {
      startVideoBtn.disabled = false;
    }
  });

  // ------------------------------------------------------------------
  // Webcam mode
  // ------------------------------------------------------------------
  async function startWebcam() {
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
    } catch (err) {
      webcamStatus.className = "banner banner-error";
      webcamStatus.textContent =
        "Could not access the webcam. Check that a camera is connected, not in use by " +
        "another application, and that camera permission is granted to this browser tab.";
      return;
    }

    webcamVideo.srcObject = webcamStream;
    webcamVideo.classList.remove("hidden");
    webcamAnnotated.classList.remove("hidden");
    startWebcamBtn.disabled = true;
    stopWebcamBtn.disabled = false;
    webcamStatus.className = "banner banner-info";
    webcamStatus.textContent = "Starting camera...";

    webcamLoopActive = true;
    webcamLoop();
  }

  function stopWebcam() {
    webcamLoopActive = false;
    if (webcamTimer) clearTimeout(webcamTimer);
    if (webcamStream) {
      webcamStream.getTracks().forEach((t) => t.stop());
      webcamStream = null;
    }
    webcamVideo.classList.add("hidden");
    webcamAnnotated.classList.add("hidden");
    startWebcamBtn.disabled = false;
    stopWebcamBtn.disabled = true;
    webcamStatus.className = "banner banner-info";
    webcamStatus.textContent = 'Webcam is stopped. Click "Start Webcam" to begin.';
  }

  async function webcamLoop() {
    if (!webcamLoopActive) return;

    if (webcamVideo.readyState >= 2) {
      const w = webcamVideo.videoWidth || 640;
      const h = webcamVideo.videoHeight || 480;
      webcamCanvas.width = w;
      webcamCanvas.height = h;
      const ctx = webcamCanvas.getContext("2d");
      ctx.drawImage(webcamVideo, 0, 0, w, h);
      const dataUrl = webcamCanvas.toDataURL("image/jpeg", 0.7);
      const { confidence, iou } = getThresholds();

      try {
        const res = await fetch("/api/detect/frame", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: dataUrl, confidence, iou }),
        });
        const data = await res.json();

        if (res.ok) {
          webcamAnnotated.src = "data:image/jpeg;base64," + data.annotated_image;

          const fireFound = data.detections.some((d) => d.class_name.toLowerCase() === "fire");
          const smokeFound = data.detections.some((d) => d.class_name.toLowerCase() === "smoke");

          if (fireFound) {
            webcamStatus.className = "banner banner-error";
            webcamStatus.textContent = "🔥 FIRE DETECTED!";
          } else if (smokeFound) {
            webcamStatus.className = "banner banner-warning";
            webcamStatus.textContent = "⚠️ SMOKE DETECTED!";
          } else {
            webcamStatus.className = "banner banner-success";
            webcamStatus.textContent = "No fire or smoke detected.";
          }

          statCurrentStatus.textContent = fireFound || smokeFound ? "Fire/Smoke Detected" : "Clear";

          if (data.new_alerts && data.new_alerts.length) {
            playAlarmIfEnabled(data.new_alerts);
            await refreshHistoryAndStats();
          }
        }
      } catch (err) {
        // Transient network hiccups shouldn't stop the live loop.
      }
    }

    webcamTimer = setTimeout(webcamLoop, 400);
  }

  startWebcamBtn.addEventListener("click", startWebcam);
  stopWebcamBtn.addEventListener("click", stopWebcam);

  // ------------------------------------------------------------------
  // Alert status + history + dashboard cards
  // ------------------------------------------------------------------
  async function refreshHistoryAndStats() {
    try {
      const res = await fetch("/api/history");
      const rows = await res.json();
      renderHistory(rows);
      renderDashboardStats(rows);
      renderAlertStatus(rows);
    } catch (err) {
      // Non-fatal; history/dashboard simply won't update this cycle.
    }
  }

  function renderHistory(rows) {
    if (!rows.length) {
      historyEmpty.classList.remove("hidden");
      historyTable.classList.add("hidden");
      return;
    }
    historyEmpty.classList.add("hidden");
    historyTable.classList.remove("hidden");

    const sorted = [...rows].sort((a, b) => (a.timestamp < b.timestamp ? 1 : -1));
    historyTableBody.innerHTML = sorted
      .map(
        (r) => `<tr>
          <td>${escapeHtml(r.timestamp)}</td>
          <td>${escapeHtml(r.detection_type)}</td>
          <td>${Number(r.confidence).toFixed(1)}%</td>
          <td>${escapeHtml(r.source)}</td>
          <td>${escapeHtml(r.alert_image_path || "-")}</td>
        </tr>`
      )
      .join("");
  }

  function renderDashboardStats(rows) {
    const fireCount = rows.filter((r) => r.detection_type.toLowerCase() === "fire").length;
    const smokeCount = rows.filter((r) => r.detection_type.toLowerCase() === "smoke").length;
    const maxConf = rows.reduce((m, r) => Math.max(m, Number(r.confidence) || 0), 0);

    statFireCount.textContent = fireCount;
    statSmokeCount.textContent = smokeCount;
    statTotalAlerts.textContent = rows.length;
    statMaxConfidence.textContent = maxConf.toFixed(1) + "%";
  }

  function renderAlertStatus(rows) {
    if (!rows.length) {
      currentAlert.className = "banner banner-success";
      currentAlert.textContent = "No active alert";
      latestDetection.textContent = "- (0.0%)";
      latestTimestamp.textContent = "-";
      return;
    }
    const sorted = [...rows].sort((a, b) => (a.timestamp < b.timestamp ? 1 : -1));
    const latest = sorted[0];
    currentAlert.className = "banner banner-error";
    currentAlert.textContent = `${latest.detection_type} detected!`;
    latestDetection.textContent = `${latest.detection_type} (${Number(latest.confidence).toFixed(1)}%)`;
    latestTimestamp.textContent = latest.timestamp;
  }

  clearHistoryBtn.addEventListener("click", async () => {
    if (!confirm("Clear all detection history? This cannot be undone.")) return;
    await fetch("/api/history/clear", { method: "POST" });
    await refreshHistoryAndStats();
    statCurrentStatus.textContent = "Idle";
  });

  // ------------------------------------------------------------------
  // Init
  // ------------------------------------------------------------------
  checkModelStatus();
  refreshHistoryAndStats();
})();
