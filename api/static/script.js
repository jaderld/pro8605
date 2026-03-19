
let mediaRecorder;
let audioChunks = [];
const recordBtn = document.getElementById('recordBtn');
const btnLabel = document.getElementById('btnLabel');
const statusTxt = document.getElementById('status');
const loader = document.getElementById('loader');
const resultsDiv = document.getElementById('results');

// ─── Interview context (filled after question generation) ───
let interviewContext = { question: '', domain: '', position: '' };

// ─── Interview form: generate question via LLM ───
const interviewForm = document.getElementById('interviewForm');
const generateBtn = document.getElementById('generateBtn');
const questionText = document.getElementById('questionText');
const questionPlaceholder = document.getElementById('questionPlaceholder');
const questionHint = document.getElementById('questionHint');

if (interviewForm) {
    interviewForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const domain = document.getElementById('fieldDomain').value.trim();
        const position = document.getElementById('fieldPosition').value.trim();
        const focus = document.getElementById('fieldFocus').value.trim();

        if (!domain || !position) return;

        generateBtn.disabled = true;
        generateBtn.textContent = 'Génération en cours…';

        try {
            const formData = new FormData();
            formData.append('domain', domain);
            formData.append('position', position);
            formData.append('focus_points', focus);

            const resp = await fetch('/generate_question/', { method: 'POST', body: formData });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.error || 'Erreur serveur');
            }
            const data = await resp.json();

            // Save context for later
            interviewContext = { question: data.question, domain, position };

            // Show question
            questionPlaceholder.style.display = 'none';
            questionText.textContent = data.question;
            questionText.classList.add('visible');
            questionHint.classList.add('visible');

            // Show recorder widget and enable recording
            const recorderWidget = document.getElementById('recorderWidget');
            if (recorderWidget) recorderWidget.classList.add('visible');
            recordBtn.disabled = false;
            statusTxt.textContent = 'Lisez la question puis cliquez pour répondre';

        } catch (err) {
            statusTxt.textContent = 'Erreur : ' + err.message;
        } finally {
            generateBtn.disabled = false;
            generateBtn.textContent = 'Générer une question d\'entretien';
        }
    });
}

recordBtn.addEventListener('click', async () => {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
            mediaRecorder.onstop = sendAudioToAPI;

            mediaRecorder.start();
            recordBtn.classList.add('stop');
            recordBtn.classList.remove('start');
            const rw = document.getElementById('recorderWidget');
            if (rw) rw.classList.add('recording');
            btnLabel.textContent = "Arrêter l'enregistrement";
            statusTxt.textContent = "Enregistrement en cours...";
            loader.classList.remove('hidden');
            resultsDiv.classList.add('hidden');
        } catch (err) {
            statusTxt.textContent = "Erreur accès au micro : " + err.message;
        }
    } else if (mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        recordBtn.classList.remove('stop');
        recordBtn.classList.add('start');
        const rw = document.getElementById('recorderWidget');
        if (rw) rw.classList.remove('recording');
        btnLabel.textContent = "Démarrer l'analyse";
        statusTxt.textContent = "L'IA analyse votre voix... ⏳";
        loader.classList.remove('hidden');
    }
});

async function sendAudioToAPI() {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append("file", audioBlob, "recording.wav");

    // Pass interview context if available
    if (interviewContext.question) {
        formData.append("interview_question", interviewContext.question);
        formData.append("interview_domain", interviewContext.domain);
        formData.append("interview_position", interviewContext.position);
    }

    // Désactiver les boutons copier/télécharger pendant le streaming
    const copyBtn = document.getElementById('copyReport');
    const dlBtn = document.getElementById('downloadReport');
    const llmReport = document.getElementById('llmReport');
    if (copyBtn) copyBtn.classList.add('btn-disabled');
    if (dlBtn) dlBtn.classList.add('btn-disabled');
    if (llmReport) llmReport.textContent = '';

    try {
        const response = await fetch('/analyze_stream/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Erreur serveur lors de l'analyse");

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let reportText = '';
        let scoresDisplayed = false;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // garder le fragment incomplet

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const jsonStr = line.slice(6);
                let event;
                try { event = JSON.parse(jsonStr); } catch { continue; }

                if (event.type === 'scores') {
                    displayScores(event.data);
                    scoresDisplayed = true;
                    statusTxt.textContent = "Rédaction du rapport en cours...";
                    // Afficher le curseur de saisie
                    if (llmReport) llmReport.classList.add('typing');
                }
                else if (event.type === 'report_line') {
                    reportText += (reportText ? '\n' : '') + event.text;
                    if (llmReport) llmReport.textContent = reportText;
                    llmReport.scrollTop = llmReport.scrollHeight;
                }
                else if (event.type === 'llm_token') {
                    reportText += event.text;
                    if (llmReport) llmReport.textContent = reportText;
                    llmReport.scrollTop = llmReport.scrollHeight;
                }
                else if (event.type === 'relevance_token') {
                    reportText += event.text;
                    if (llmReport) llmReport.textContent = reportText;
                    llmReport.scrollTop = llmReport.scrollHeight;
                }
                else if (event.type === 'done') {
                    // Rapport terminé : activer les boutons
                    if (llmReport) llmReport.classList.remove('typing');
                    if (copyBtn) copyBtn.classList.remove('btn-disabled');
                    if (dlBtn) dlBtn.classList.remove('btn-disabled');
                    statusTxt.textContent = "Analyse terminée avec succès !";
                }
                else if (event.type === 'error') {
                    throw new Error(event.message || "Erreur serveur");
                }
            }
        }

        loader.classList.add('hidden');
    } catch (error) {
        console.error(error);
        statusTxt.textContent = "Erreur : " + error.message;
        loader.classList.add('hidden');
        // Réactiver les boutons en cas d'erreur
        if (copyBtn) copyBtn.classList.remove('btn-disabled');
        if (dlBtn) dlBtn.classList.remove('btn-disabled');
        if (llmReport) llmReport.classList.remove('typing');
    }
}

function displayScores(data) {
    resultsDiv.classList.remove('hidden');

    // Mise à jour sécurisée des éléments
    const update = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    };
    // 1. Score & Interpretation
    update('globalScore', (data.final_score || 0) + "/100");
    update('interpretation', data.interpretation || "Analyse en cours");

    // 2. Transcription & Sentiment
    const textData = data.details?.text_analysis || {};
    update('transcriptionText', textData.transcription || "Pas de texte détecté.");
    update('sentimentVal', textData.sentiment || "Neutre 😐");

    // 3. Audio (Physique)
    const audioData = data.details?.audio_analysis || {};
    update('volVal', audioData.volume || "--");
    update('tempoVal', audioData.tempo_bpm || "--");
    update('pauseVal', audioData.pause_ratio || "--");

    // 4. Émotion (IA)
    const emoData = data.details?.emotion_analysis || {};
    update('emotionLabel', emoData.label || "Inconnu");
    update('confidenceVal', emoData.confidence || "0%");

    // 5. Tics de Langage (Liste dynamique)
    const list = document.getElementById('fillersList');
    if (list) {
        list.innerHTML = '';
        const fillers = textData.fillers || {};
        if (Object.keys(fillers).length === 0) {
            list.innerHTML = '<li>Aucun tic détecté ✅</li>';
        } else {
            for (const [word, count] of Object.entries(fillers)) {
                const li = document.createElement('li');
                li.innerHTML = `<strong>${word}</strong>: ${count}`;
                list.appendChild(li);
            }
        }
    }

    // 6. Boutons Copier / Télécharger le rapport (attachés une seule fois)
    const copyBtn = document.getElementById('copyReport');
    if (copyBtn && !copyBtn._bound) {
        copyBtn._bound = true;
        copyBtn.onclick = () => {
            const llmReport = document.getElementById('llmReport');
            const text = llmReport ? llmReport.textContent : '';
            navigator.clipboard.writeText(text).then(() => {
                copyBtn.textContent = 'Copié ✓';
                setTimeout(() => { copyBtn.textContent = 'Copier'; }, 2000);
            });
        };
    }

    const dlBtn = document.getElementById('downloadReport');
    if (dlBtn && !dlBtn._bound) {
        dlBtn._bound = true;
        dlBtn.onclick = () => {
            const llmReport = document.getElementById('llmReport');
            const text = llmReport ? llmReport.textContent : '';
            const blob = new Blob([text], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'rapport_entretien.txt';
            a.click();
            URL.revokeObjectURL(url);
        };
    }
}