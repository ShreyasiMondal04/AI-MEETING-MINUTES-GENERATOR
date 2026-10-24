/**
 * MinuteAI — Executive Frontend Controller & UI Engine
 */

document.addEventListener("DOMContentLoaded", () => {
    // Current Application State
    const state = {
        currentMeetingId: null,
        currentMinutes: null,
        currentTranscript: [],
        currentAudioUrl: null,
        isProcessing: false,
    };

    // DOM Elements
    const themeToggle = document.getElementById("themeToggle");
    const systemStatus = document.getElementById("systemStatus");
    const statusText = document.getElementById("statusText");

    const uploadForm = document.getElementById("uploadForm");
    const audioFileInput = document.getElementById("audioFile");
    const dropZone = document.getElementById("dropZone");
    const fileChosenBar = document.getElementById("fileChosenBar");
    const selectedFileName = document.getElementById("selectedFileName");
    const selectedFileSize = document.getElementById("selectedFileSize");
    const btnClearFile = document.getElementById("btnClearFile");
    const generateBtn = document.getElementById("generateBtn");

    const settingsToggle = document.getElementById("settingsToggle");
    const settingsContent = document.getElementById("settingsContent");
    const settingsDrawer = document.querySelector(".settings-drawer");

    const stepperSection = document.getElementById("stepperSection");
    const stepperMessage = document.getElementById("stepperMessage");
    const stepperPercent = document.getElementById("stepperPercent");
    const progressBarFill = document.getElementById("progressBarFill");

    const resultsSection = document.getElementById("resultsSection");
    const meetingAudioPlayer = document.getElementById("meetingAudioPlayer");
    const currentMeetingTitle = document.getElementById("currentMeetingTitle");

    // Tab Buttons
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanels = document.querySelectorAll(".tab-panel");

    // Minutes Fields
    const metaDate = document.getElementById("metaDate");
    const metaDuration = document.getElementById("metaDuration");
    const metaAttendees = document.getElementById("metaAttendees");
    const summaryText = document.getElementById("summaryText");
    const decisionsList = document.getElementById("decisionsList");
    const actionItemsTbody = document.getElementById("actionItemsTbody");
    const topicsGrid = document.getElementById("topicsGrid");
    const noteText = document.getElementById("noteText");

    // Transcript Fields
    const dialogueFeed = document.getElementById("dialogueFeed");
    const transcriptSearch = document.getElementById("transcriptSearch");
    const speakerFilter = document.getElementById("speakerFilter");
    const transcriptCountBadge = document.getElementById("transcriptCountBadge");
    const btnCopyTranscript = document.getElementById("btnCopyTranscript");

    // Analytics Fields
    const talkTimeBar = document.getElementById("talkTimeBar");
    const speakerAnalyticsGrid = document.getElementById("speakerAnalyticsGrid");

    // Evaluation Fields
    const werValue = document.getElementById("werValue");
    const werSub = document.getElementById("werSub");
    const derValue = document.getElementById("derValue");
    const derSub = document.getElementById("derSub");
    const rouge1Value = document.getElementById("rouge1Value");
    const rougeSub = document.getElementById("rougeSub");
    const bleuValue = document.getElementById("bleuValue");
    const bleuSub = document.getElementById("bleuSub");
    const btnRunEval = document.getElementById("btnRunEval");

    // Download Links
    const btnDownloadPdf = document.getElementById("btnDownloadPdf");
    const btnDownloadDocx = document.getElementById("btnDownloadDocx");
    const btnDownloadMd = document.getElementById("btnDownloadMd");
    const btnDownloadTxt = document.getElementById("btnDownloadTxt");
    const btnDownloadJson = document.getElementById("btnDownloadJson");

    // Rename Modal Elements
    const renameModal = document.getElementById("renameModal");
    const btnOpenRenameModal = document.getElementById("btnOpenRenameModal");
    const btnCloseRenameModal = document.getElementById("btnCloseRenameModal");
    const btnCancelRename = document.getElementById("btnCancelRename");
    const btnSaveRename = document.getElementById("btnSaveRename");
    const renameFieldsList = document.getElementById("renameFieldsList");

    // Toast Container
    const toastContainer = document.getElementById("toastContainer");

    // =========================================================================
    // Theme Management
    // =========================================================================
    const savedTheme = localStorage.getItem("minuteai_theme") || "dark";
    if (savedTheme === "light") {
        document.body.classList.remove("dark-theme");
        document.body.classList.add("light-theme");
    }

    themeToggle.addEventListener("click", () => {
        const isLight = document.body.classList.toggle("light-theme");
        document.body.classList.toggle("dark-theme", !isLight);
        localStorage.setItem("minuteai_theme", isLight ? "light" : "dark");
    });

    // =========================================================================
    // Toast Notification System
    // =========================================================================
    function showToast(message, type = "info") {
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateY(10px)";
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    // =========================================================================
    // System Health Check
    // =========================================================================
    async function checkHealth() {
        try {
            const res = await fetch("/health");
            if (res.ok) {
                const data = await res.json();
                statusText.textContent = `Online • ${data.device}`;
            }
        } catch {
            statusText.textContent = "Offline / Server Error";
            document.querySelector(".status-dot").style.backgroundColor = "var(--danger)";
        }
    }
    checkHealth();

    // =========================================================================
    // Settings Drawer Toggle
    // =========================================================================
    settingsToggle.addEventListener("click", () => {
        const isClosed = settingsContent.hidden;
        settingsContent.hidden = !isClosed;
        settingsDrawer.classList.toggle("open", isClosed);
    });

    // =========================================================================
    // File Upload & Drag-and-Drop Handling
    // =========================================================================
    function handleFileSelection(file) {
        if (!file) return;

        const allowedExts = [".wav", ".mp3", ".mpeg", ".m4a", ".flac", ".ogg", ".aac"];
        const ext = "." + file.name.split(".").pop().toLowerCase();

        if (!allowedExts.includes(ext)) {
            showToast(`Unsupported format '${ext}'. Please choose a WAV, MP3, MPEG, or M4A file.`, "error");
            audioFileInput.value = "";
            return;
        }

        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        selectedFileName.textContent = file.name;
        selectedFileSize.textContent = `${sizeMB} MB`;
        fileChosenBar.hidden = false;
        dropZone.querySelector(".drop-zone-content").style.display = "none";
    }

    audioFileInput.addEventListener("change", (e) => {
        handleFileSelection(e.target.files[0]);
    });

    btnClearFile.addEventListener("click", (e) => {
        e.stopPropagation();
        audioFileInput.value = "";
        fileChosenBar.hidden = true;
        dropZone.querySelector(".drop-zone-content").style.display = "block";
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length) {
            audioFileInput.files = e.dataTransfer.files;
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    // =========================================================================
    // Progress Stepper Simulation
    // =========================================================================
    let progressTimer = null;

    function startProgressStepper() {
        stepperSection.hidden = false;
        resultsSection.hidden = true;
        generateBtn.disabled = true;

        const stages = [
            { stage: 1, name: "Audio Preprocessing & Normalization...", pct: 15 },
            { stage: 2, name: "Running Whisper Speech-to-Text...", pct: 40 },
            { stage: 3, name: "Running Speaker Diarization...", pct: 65 },
            { stage: 4, name: "Extracting NLP Topics & Action Items...", pct: 85 },
            { stage: 5, name: "Generating Multi-Format Deliverables...", pct: 95 }
        ];

        let currentIdx = 0;
        updateStepperStage(stages[0]);

        progressTimer = setInterval(() => {
            if (currentIdx < stages.length - 1) {
                currentIdx++;
                updateStepperStage(stages[currentIdx]);
            }
        }, 3500);
    }

    function updateStepperStage(stageObj) {
        stepperMessage.textContent = stageObj.name;
        stepperPercent.textContent = `${stageObj.pct}%`;
        progressBarFill.style.width = `${stageObj.pct}%`;

        for (let i = 1; i <= 5; i++) {
            const el = document.getElementById(`stage${i}`);
            if (i < stageObj.stage) {
                el.className = "stage-step completed";
            } else if (i === stageObj.stage) {
                el.className = "stage-step active";
            } else {
                el.className = "stage-step";
            }
        }
    }

    function finishProgressStepper() {
        clearInterval(progressTimer);
        stepperPercent.textContent = "100%";
        progressBarFill.style.width = "100%";
        for (let i = 1; i <= 5; i++) {
            document.getElementById(`stage${i}`).className = "stage-step completed";
        }
        setTimeout(() => {
            stepperSection.hidden = true;
            generateBtn.disabled = false;
        }, 600);
    }

    // =========================================================================
    // Form Submission: Upload Audio
    // =========================================================================
    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const file = audioFileInput.files[0];
        if (!file) {
            showToast("Please select an audio file to process.", "error");
            return;
        }

        const formData = new FormData(uploadForm);
        startProgressStepper();

        try {
            const response = await fetch("/api/process-meeting", {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.detail || "Audio processing failed.");
            }

            finishProgressStepper();
            displayAllResults(data);
            showToast("Meeting minutes generated successfully!", "success");

        } catch (err) {
            clearInterval(progressTimer);
            stepperSection.hidden = true;
            generateBtn.disabled = false;
            showToast(`Processing Error: ${err.message}`, "error");
        }
    });

    // =========================================================================
    // Quick Demo Sample Loader
    // =========================================================================
    document.querySelectorAll(".btn-sample-load").forEach((btn) => {
        btn.addEventListener("click", async () => {
            const sampleId = btn.getAttribute("data-sample-id");
            btn.disabled = true;
            startProgressStepper();

            try {
                const response = await fetch(`/api/process-sample/${sampleId}`, {
                    method: "POST"
                });
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.detail || "Failed to load sample.");
                }

                finishProgressStepper();
                displayAllResults(data);
                showToast(`Loaded Sample Meeting '${sampleId}'!`, "success");

            } catch (err) {
                clearInterval(progressTimer);
                stepperSection.hidden = true;
                showToast(`Sample Error: ${err.message}`, "error");
            } finally {
                btn.disabled = false;
            }
        });
    });

    // =========================================================================
    // Display Results in Dashboard
    // =========================================================================
    function displayAllResults(data) {
        state.currentMeetingId = data.meeting_id;
        state.currentMinutes = data.minutes;
        state.currentTranscript = data.speaker_transcript || [];
        state.currentAudioUrl = data.audio_url || `/api/audio/${data.meeting_id}.wav`;

        const minutes = data.minutes;

        // Set Audio Player Source
        if (state.currentAudioUrl) {
            meetingAudioPlayer.src = state.currentAudioUrl;
            meetingAudioPlayer.load();
        }

        currentMeetingTitle.textContent = minutes.title || "Meeting Recording";

        // Tab 1: Metadata
        metaDate.textContent = minutes.date || new Date().toLocaleDateString();
        metaDuration.textContent = minutes.duration_formatted || "N/A";

        metaAttendees.replaceChildren();
        (minutes.participants || []).forEach((name) => {
            const pill = document.createElement("span");
            pill.className = "attendee-pill";
            pill.textContent = name;
            metaAttendees.appendChild(pill);
        });

        // Summary
        summaryText.textContent = minutes.summary || "No summary available.";

        // Decisions
        decisionsList.replaceChildren();
        if (minutes.decisions && minutes.decisions.length) {
            minutes.decisions.forEach((dec) => {
                const li = document.createElement("li");
                li.textContent = dec;
                decisionsList.appendChild(li);
            });
        } else {
            const li = document.createElement("li");
            li.textContent = "No explicit decisions identified.";
            decisionsList.appendChild(li);
        }

        // Action Items Table
        actionItemsTbody.replaceChildren();
        if (minutes.action_items && minutes.action_items.length) {
            minutes.action_items.forEach((item, idx) => {
                const tr = document.createElement("tr");

                // Checkbox
                const tdCheck = document.createElement("td");
                const checkbox = document.createElement("input");
                checkbox.type = "checkbox";
                checkbox.className = "task-checkbox";
                checkbox.checked = item.status === "Completed";
                checkbox.addEventListener("change", () => {
                    tr.style.opacity = checkbox.checked ? "0.45" : "1";
                    tr.style.textDecoration = checkbox.checked ? "line-through" : "none";
                });
                tdCheck.appendChild(checkbox);

                // Task
                const tdTask = document.createElement("td");
                tdTask.innerHTML = `<strong>${item.task}</strong>`;

                // Assignee
                const tdOwner = document.createElement("td");
                tdOwner.innerHTML = `<span class="attendee-pill">${item.responsible_person || "Team"}</span>`;

                // Deadline
                const tdDeadline = document.createElement("td");
                tdDeadline.innerHTML = `<span class="deadline-pill">⏱ ${item.deadline || "Not specified"}</span>`;

                // Priority
                const tdPriority = document.createElement("td");
                const prio = (item.priority || "Normal").toLowerCase();
                tdPriority.innerHTML = `<span class="priority-badge priority-${prio}">${item.priority || "Normal"}</span>`;

                tr.appendChild(tdCheck);
                tr.appendChild(tdTask);
                tr.appendChild(tdOwner);
                tr.appendChild(tdDeadline);
                tr.appendChild(tdPriority);
                actionItemsTbody.appendChild(tr);
            });
        } else {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td colspan="5" style="text-align: center; color: var(--text-muted); padding: 18px;">No action items extracted.</td>`;
            actionItemsTbody.appendChild(tr);
        }

        // Topics Grid
        topicsGrid.replaceChildren();
        if (minutes.topics && minutes.topics.length) {
            minutes.topics.forEach((top) => {
                const card = document.createElement("div");
                card.className = "topic-card";

                const title = document.createElement("div");
                title.className = "topic-title";
                title.textContent = top.topic;

                const ul = document.createElement("ul");
                ul.className = "topic-points-list";
                (top.points || []).forEach((pt) => {
                    const li = document.createElement("li");
                    li.textContent = pt;
                    ul.appendChild(li);
                });

                card.appendChild(title);
                card.appendChild(ul);
                topicsGrid.appendChild(card);
            });
        }

        // Notes
        noteText.textContent = minutes.note || "Automated AI Minutes Synthesis.";

        // Tab 2: Diarized Dialogue Feed
        renderDialogueFeed(state.currentTranscript);
        populateSpeakerFilter(state.currentTranscript);

        // Tab 3: Speaker Participation Analytics
        renderAnalytics(minutes.speaker_analytics);

        // Tab 4: Evaluation Benchmark (Fetch auto)
        fetchEvaluationMetrics(data.meeting_id);

        // Tab 5: Setup Download Links
        setupExportLinks(data.meeting_id, data.export_urls);

        // Reveal Results Dashboard
        resultsSection.hidden = false;
        resultsSection.scrollIntoView({ behavior: "smooth" });
    }

    // =========================================================================
    // Tab 2: Dialogue Feed Renderer
    // =========================================================================
    function renderDialogueFeed(segments) {
        dialogueFeed.replaceChildren();
        transcriptCountBadge.textContent = segments.length;

        if (!segments || segments.length === 0) {
            dialogueFeed.innerHTML = `<p style="text-align: center; color: var(--text-muted); padding: 20px;">No transcript available.</p>`;
            return;
        }

        segments.forEach((turn, idx) => {
            const card = document.createElement("div");
            card.className = "turn-card";
            card.setAttribute("data-speaker", turn.speaker);

            // Avatar Color Determinant
            const spkNum = parseInt((turn.speaker || "").replace(/\D/g, "")) || (idx % 4) + 1;
            const avatarClass = `avatar-spk-${((spkNum - 1) % 4) + 1}`;

            // Avatar
            const avatar = document.createElement("div");
            avatar.className = `speaker-avatar-circle ${avatarClass}`;
            avatar.textContent = (turn.speaker || "S").substring(0, 2).toUpperCase();

            // Content
            const content = document.createElement("div");
            content.className = "turn-content";

            const metaRow = document.createElement("div");
            metaRow.className = "turn-meta-row";

            const name = document.createElement("span");
            name.className = "turn-speaker-name";
            name.textContent = turn.speaker || "Speaker";

            // Clickable timestamp button that seeks audio!
            const tsBtn = document.createElement("button");
            tsBtn.type = "button";
            tsBtn.className = "turn-timestamp-btn";
            tsBtn.textContent = turn.timestamp || `[${turn.start || 0}s]`;
            tsBtn.title = "Click to jump audio here";
            tsBtn.addEventListener("click", () => {
                if (meetingAudioPlayer && turn.start !== undefined) {
                    meetingAudioPlayer.currentTime = turn.start;
                    meetingAudioPlayer.play();
                }
            });

            metaRow.appendChild(name);
            metaRow.appendChild(tsBtn);

            const text = document.createElement("p");
            text.className = "turn-text";
            text.textContent = turn.text;

            content.appendChild(metaRow);
            content.appendChild(text);

            card.appendChild(avatar);
            card.appendChild(content);
            dialogueFeed.appendChild(card);
        });
    }

    function populateSpeakerFilter(segments) {
        const speakers = [...new Set(segments.map((s) => s.speaker))];
        speakerFilter.replaceChildren();

        const optAll = document.createElement("option");
        optAll.value = "all";
        optAll.textContent = "All Speakers";
        speakerFilter.appendChild(optAll);

        speakers.forEach((spk) => {
            const opt = document.createElement("option");
            opt.value = spk;
            opt.textContent = spk;
            speakerFilter.appendChild(opt);
        });
    }

    // Filter and Search
    function filterTranscript() {
        const query = transcriptSearch.value.toLowerCase().trim();
        const selectedSpk = speakerFilter.value;
        const cards = dialogueFeed.querySelectorAll(".turn-card");

        cards.forEach((card) => {
            const spk = card.getAttribute("data-speaker");
            const txt = card.querySelector(".turn-text").textContent.toLowerCase();

            const matchesSpk = selectedSpk === "all" || spk === selectedSpk;
            const matchesQuery = !query || txt.includes(query);

            card.style.display = matchesSpk && matchesQuery ? "flex" : "none";
        });
    }

    transcriptSearch.addEventListener("input", filterTranscript);
    speakerFilter.addEventListener("change", filterTranscript);

    btnCopyTranscript.addEventListener("click", () => {
        const lines = state.currentTranscript.map((t) => `${t.timestamp || ""} ${t.speaker}: ${t.text}`);
        navigator.clipboard.writeText(lines.join("\n"));
        showToast("Transcript copied to clipboard!", "success");
    });

    // =========================================================================
    // Tab 3: Speaker Participation Analytics
    // =========================================================================
    function renderAnalytics(analytics) {
        if (!analytics || !analytics.speakers) return;

        talkTimeBar.replaceChildren();
        speakerAnalyticsGrid.replaceChildren();

        analytics.speakers.forEach((s, idx) => {
            const spkNum = parseInt(s.speaker.replace(/\D/g, "")) || (idx % 4) + 1;
            const sliceClass = `slice-spk-${((spkNum - 1) % 4) + 1}`;

            // Distribution bar slice
            const slice = document.createElement("div");
            slice.className = `talk-slice ${sliceClass}`;
            slice.style.width = `${Math.max(12, s.share_percentage)}%`;
            slice.textContent = `${s.speaker}: ${s.share_percentage}%`;
            talkTimeBar.appendChild(slice);

            // Detailed card
            const card = document.createElement("div");
            card.className = "speaker-stat-card";
            card.innerHTML = `
                <div class="stat-speaker-title">${s.speaker}</div>
                <div class="stat-metric-row">
                    <span class="stat-metric-label">Speaking Time</span>
                    <span class="stat-metric-val">${s.talk_time_sec}s (${s.share_percentage}%)</span>
                </div>
                <div class="stat-metric-row">
                    <span class="stat-metric-label">Vocal Turns</span>
                    <span class="stat-metric-val">${s.turns}</span>
                </div>
                <div class="stat-metric-row">
                    <span class="stat-metric-label">Total Words Spoken</span>
                    <span class="stat-metric-val">${s.word_count} words</span>
                </div>
                <div class="stat-metric-row">
                    <span class="stat-metric-label">Speaking Pace</span>
                    <span class="stat-metric-val">${s.talk_time_sec > 0 ? Math.round((s.word_count / s.talk_time_sec) * 60) : 0} WPM</span>
                </div>
            `;
            speakerAnalyticsGrid.appendChild(card);
        });
    }

    // =========================================================================
    // Tab 4: Evaluation Benchmark API
    // =========================================================================
    async function fetchEvaluationMetrics(meetingId) {
        try {
            const res = await fetch(`/api/evaluate/${meetingId}`);
            if (!res.ok) return;

            const data = await res.json();
            if (data.status === "success" && data.metrics) {
                const m = data.metrics;
                werValue.textContent = m.speech_to_text.wer_percentage;
                werSub.textContent = `Word Accuracy: ${m.speech_to_text.word_accuracy}`;

                derValue.textContent = m.speaker_diarization.turn_attribution_accuracy;
                derSub.textContent = `Time-Weighted Acc: ${m.speaker_diarization.time_weighted_accuracy}`;

                rouge1Value.textContent = m.summarization.rouge_1_f1;
                rougeSub.textContent = `ROUGE-L F1: ${m.summarization.rouge_l_f1}`;

                bleuValue.textContent = m.summarization.bleu_score;
                bleuSub.textContent = `ROUGE-2 F1: ${m.summarization.rouge_2_f1}`;
            } else {
                werValue.textContent = "N/A";
                werSub.textContent = "Reference file not provided";
                derValue.textContent = "N/A";
                derSub.textContent = "Reference file not provided";
                rouge1Value.textContent = "N/A";
                rougeSub.textContent = "Reference file not provided";
                bleuValue.textContent = "N/A";
                bleuSub.textContent = "Reference file not provided";
            }
        } catch {
            // Silently ignore if evaluation is unavailable
        }
    }

    btnRunEval.addEventListener("click", () => {
        if (state.currentMeetingId) {
            fetchEvaluationMetrics(state.currentMeetingId);
            showToast("Evaluation metrics recalculated!", "info");
        }
    });

    // =========================================================================
    // Tab 5: Export Links Setup
    // =========================================================================
    function setupExportLinks(meetingId, urls) {
        btnDownloadPdf.href = `/api/export/${meetingId}/pdf`;
        btnDownloadDocx.href = `/api/export/${meetingId}/docx`;
        btnDownloadMd.href = `/api/export/${meetingId}/md`;
        btnDownloadTxt.href = `/api/export/${meetingId}/txt`;
        btnDownloadJson.href = `/api/export/${meetingId}/json`;
    }

    // =========================================================================
    // Tab Navigation
    // =========================================================================
    tabBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-tab");

            tabBtns.forEach((b) => b.classList.remove("active"));
            tabPanels.forEach((p) => p.classList.remove("active"));

            btn.classList.add("active");
            document.getElementById(targetId).classList.add("active");
        });
    });

    // =========================================================================
    // Rename Speakers Modal
    // =========================================================================
    btnOpenRenameModal.addEventListener("click", () => {
        if (!state.currentTranscript.length) return;

        const uniqueSpeakers = [...new Set(state.currentTranscript.map((t) => t.speaker))];
        renameFieldsList.replaceChildren();

        uniqueSpeakers.forEach((spk) => {
            const row = document.createElement("div");
            row.className = "form-group";
            row.innerHTML = `
                <label for="rename_${spk}">Original Label: <strong>${spk}</strong></label>
                <input type="text" id="rename_${spk}" data-original="${spk}" value="${spk}" class="custom-input">
            `;
            renameFieldsList.appendChild(row);
        });

        renameModal.hidden = false;
    });

    function closeRenameModal() {
        renameModal.hidden = true;
    }

    btnCloseRenameModal.addEventListener("click", closeRenameModal);
    btnCancelRename.addEventListener("click", closeRenameModal);

    btnSaveRename.addEventListener("click", async () => {
        const inputs = renameFieldsList.querySelectorAll("input");
        const mapping = {};

        inputs.forEach((input) => {
            const original = input.getAttribute("data-original");
            const newName = input.value.trim();
            if (newName && newName !== original) {
                mapping[original] = newName;
            }
        });

        if (Object.keys(mapping).length === 0) {
            closeRenameModal();
            return;
        }

        try {
            const res = await fetch(`/api/rename-speakers/${state.currentMeetingId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mapping })
            });

            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Renaming failed");

            closeRenameModal();
            displayAllResults({
                meeting_id: state.currentMeetingId,
                minutes: data.minutes,
                speaker_transcript: data.speaker_transcript,
                audio_url: state.currentAudioUrl
            });
            showToast("Speaker names updated across all minutes!", "success");

        } catch (err) {
            showToast(`Error renaming: ${err.message}`, "error");
        }
    });
});
