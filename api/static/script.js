
let mediaRecorder;
let audioChunks = [];
const recordBtn = document.getElementById('recordBtn');
const btnLabel = document.getElementById('btnLabel');
const statusTxt = document.getElementById('status');
const loader = document.getElementById('loader');
const resultsDiv = document.getElementById('results');

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
        btnLabel.textContent = "Démarrer l'analyse";
        statusTxt.textContent = "L'IA analyse votre voix... ⏳";
        loader.classList.remove('hidden');
    }
});

async function sendAudioToAPI() {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append("file", audioBlob, "recording.wav");

    try {
        const response = await fetch('/analyze_file/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Erreur serveur lors de l'analyse");

        const data = await response.json();
        displayResults(data);
        statusTxt.textContent = "Analyse terminée avec succès !";
        loader.classList.add('hidden');
    } catch (error) {
        console.error(error);
        statusTxt.textContent = "Erreur : " + error.message;
        loader.classList.add('hidden');
    }
}

function displayResults(data) {
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
    update('tempoVal', (audioData.tempo_bpm || 0) + " BPM");
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

    // 6. Affichage automatique du rapport LLM
    const llmReport = document.getElementById('llmReport');
    if (llmReport) {
        llmReport.textContent = data.llm_report || 'Rapport LLM non disponible.';
    }
}