let mediaRecorder;
let audioChunks = [];
const recordBtn = document.getElementById('recordBtn');
const statusTxt = document.getElementById('status');
const resultsDiv = document.getElementById('results');

recordBtn.addEventListener('click', async () => {
    if (recordBtn.textContent.includes('Démarrer')) {
        // DÉMARRER L'ENREGISTREMENT
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
        
        mediaRecorder.onstop = sendAudioToAPI;

        mediaRecorder.start();
        recordBtn.textContent = "🛑 Arrêter l'Enregistrement";
        recordBtn.classList.remove('start');
        recordBtn.classList.add('stop');
        statusTxt.textContent = "Enregistrement en cours...";
        resultsDiv.classList.add('hidden');
    } else {
        // ARRÊTER
        mediaRecorder.stop();
        recordBtn.textContent = "🎤 Démarrer l'Enregistrement";
        recordBtn.classList.remove('stop');
        recordBtn.classList.add('start');
        statusTxt.textContent = "Analyse en cours... (Cela peut prendre quelques secondes)";
    }
});

async function sendAudioToAPI() {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append("file", audioBlob, "recording.wav");

    try {
        // Appel à ton API FastAPI
        const response = await fetch('/analyze_file/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Erreur API");

        const data = await response.json();
        displayResults(data);
        statusTxt.textContent = "Analyse terminée !";

    } catch (error) {
        console.error(error);
        statusTxt.textContent = "Erreur lors de l'analyse : " + error.message;
    }
}

function displayResults(data) {
    resultsDiv.classList.remove('hidden');

    // 1. Score Global (depuis ML Model)
    // Adapte selon la structure exacte de ta réponse JSON
    const score = data.final_scoring?.overall_score || data.global_score || "N/A";
    document.getElementById('globalScore').textContent = score + "/100";

    // 2. Transcription
    document.getElementById('transcriptionText').textContent = data.transcription || "Pas de texte détecté.";

    // 3. NLP & Émotion
    document.getElementById('sentimentVal').textContent = data.nlp?.sentiment_score || "0";
    // Si tu as un champ stress dans acoustics ou emotion_analysis
    document.getElementById('stressVal').textContent = data.acoustics?.pause_ratio ? (data.acoustics.pause_ratio * 100).toFixed(0) + "% (Pauses)" : "N/A";

    // 4. Liste des Tics
    const fillersList = document.getElementById('fillersList');
    fillersList.innerHTML = '';
    const fillers = data.nlp?.fillers_details || {};
    
    if (Object.keys(fillers).length === 0) {
        fillersList.innerHTML = '<li>Aucun tic détecté ✅</li>';
    } else {
        for (const [word, count] of Object.entries(fillers)) {
            const li = document.createElement('li');
            li.textContent = `${word}: ${count}`;
            fillersList.appendChild(li);
        }
    }
}