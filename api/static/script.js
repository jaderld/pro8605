let mediaRecorder;
let audioChunks = [];
const recordBtn = document.getElementById('recordBtn');
const statusTxt = document.getElementById('status');
const resultsDiv = document.getElementById('results');

recordBtn.addEventListener('click', async () => {
    if (recordBtn.textContent.includes('Démarrer')) {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
        mediaRecorder.onstop = sendAudioToAPI;

        mediaRecorder.start();
        recordBtn.textContent = "🛑 Arrêter l'Enregistrement";
        recordBtn.classList.replace('start', 'stop');
        statusTxt.textContent = "Enregistrement en cours...";
        resultsDiv.classList.add('hidden');
    } else {
        mediaRecorder.stop();
        recordBtn.textContent = "🎤 Démarrer l'Enregistrement";
        recordBtn.classList.replace('stop', 'start');
        statusTxt.textContent = "L'IA analyse votre voix... ⏳";
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

    } catch (error) {
        console.error(error);
        statusTxt.textContent = "Erreur : " + error.message;
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
}