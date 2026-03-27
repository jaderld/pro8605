"""
Générateur de rapport structuré basé sur les règles métier RH.
Produit un rapport en 5 sections + une conclusion optionnelle générée par LLM.
"""


def generate_structured_report(
    score: float,
    interpretation: str,
    sentiment_label: str,
    emotion: str,
    vol_raw: float,
    vol_display: str,
    tempo_val: float,
    label_tempo: str,
    pause_ratio_pct: float,
    filler_count: int,
    fillers_dict: dict,
    transcription: str,
    word_count: int = 0,
    llm_conclusion: str | None = None,
) -> tuple[str, list[str], list[str]]:
    """
    Génère un rapport RH structuré en 5 sections à partir des métriques.
    Si llm_conclusion est fourni, une 6e section apparaît.

    Retourne :
        (rapport_complet: str, points_forts: list[str], axes: list[str])
    Les listes points_forts et axes permettent à l'appelant de les passer
    à l'OllamaClient pour construire le prompt LLM.
    """
    lines: list[str] = []
    sep = "─" * 60
    vol_pct = round(vol_raw * 1000, 1)  # passage à l'échelle d'affichage
    wc = word_count if word_count > 0 else len(transcription.split())
    sentiment_clean = (
        sentiment_label.replace("😊", "").replace("😟", "").replace("😐", "").strip()
    )
    emotion_lower = emotion.lower()

    # ══════════════════════════════════════════════════════════════
    # 1. RÉSUMÉ GLOBAL
    # ══════════════════════════════════════════════════════════════
    lines.append(sep)
    lines.append("1. RÉSUMÉ GLOBAL")
    lines.append(sep)

    if score >= 80:
        perf_msg = f"Excellente performance globale (score : {score}/100)."
        perf_detail = (
            "Le candidat démontre une aisance à l'oral et une très bonne maîtrise "
            "de l'expression en entretien. Son discours est fluide, structuré et convaincant."
        )
    elif score >= 65:
        perf_msg = f"Bonne performance globale (score : {score}/100)."
        perf_detail = (
            "Le candidat montre une aisance satisfaisante à l'oral. "
            "Quelques points mineurs peuvent encore être améliorés pour atteindre l'excellence."
        )
    elif score >= 45:
        perf_msg = f"Performance correcte mais perfectible (score : {score}/100)."
        perf_detail = (
            "L'entretien révèle des qualités intéressantes ainsi que des axes d'amélioration "
            "identifiables. Un travail ciblé permettrait des progrès sensibles."
        )
    elif score >= 25:
        perf_msg = f"Performance insuffisante (score : {score}/100)."
        perf_detail = (
            "L'entretien met en évidence plusieurs difficultés nécessitant "
            "un travail préparatoire sérieux, notamment sur la fluidité et la maîtrise du stress."
        )
    elif score >= 10:
        perf_msg = f"Performance à renforcer (score : {score}/100)."
        perf_detail = (
            "L'entretien révèle des axes d'amélioration importants, notamment sur la fluidité "
            "du discours et la présence des tics de langage. "
            "Un entraînement régulier permettra des progrès rapides."
        )
    else:
        perf_msg = f"Performance très insuffisante (score : {score}/100)."
        perf_detail = (
            "L'entretien révèle des difficultés importantes sur plusieurs dimensions : "
            "expression orale, gestion du stress et fluidité du discours. "
            "Une préparation approfondie est fortement recommandée avant tout entretien réel."
        )

    if any(k in emotion_lower for k in ("stress", "anxieux", "nerveux")):
        emotion_ctx = "Un niveau de stress élevé est perceptible dans la voix et le débit."
    elif any(k in emotion_lower for k in ("confian", "calm", "serein")):
        emotion_ctx = "Le candidat semble confiant et à l'aise."
    elif any(k in emotion_lower for k in ("neutre", "neutral")):
        emotion_ctx = "L'état émotionnel détecté est neutre, sans excès de stress ni d'enthousiasme."
    else:
        emotion_ctx = f"L'émotion dominante détectée est : {emotion}."

    lines.append(f"{perf_msg} {perf_detail}")
    lines.append(f"{emotion_ctx} Sentiment global : {sentiment_clean}.")
    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # 2. ANALYSE DES SCORES
    # ══════════════════════════════════════════════════════════════
    lines.append(sep)
    lines.append("2. ANALYSE DES SCORES")
    lines.append(sep)

    # Score global
    if score >= 80:
        lines.append(f"• Score global : {score}/100 → Excellent.")
    elif score >= 65:
        lines.append(f"• Score global : {score}/100 → Satisfaisant.")
    elif score >= 45:
        lines.append(f"• Score global : {score}/100 → Moyen. Des progrès sont possibles.")
    elif score >= 25:
        lines.append(f"• Score global : {score}/100 → Faible. Un travail de fond est nécessaire.")
    elif score >= 10:
        penalty_info = (
            f" La pénalité est principalement due aux tics de langage ({filler_count} détectés)."
            if filler_count > 3
            else ""
        )
        lines.append(f"• Score global : {score}/100 → À renforcer.{penalty_info}")
    else:
        penalty_info = (
            f" La pénalité est principalement due aux tics de langage ({filler_count} détectés)."
            if filler_count > 3
            else ""
        )
        lines.append(f"• Score global : {score}/100 → Très insuffisant.{penalty_info}")

    # Volume
    if vol_pct < 30:
        lines.append(
            f"• Volume sonore : {vol_display} → Trop faible. "
            "La voix est peu audible, ce qui peut nuire à la crédibilité et à l'assertivité."
        )
    elif vol_pct <= 70:
        lines.append(
            f"• Volume sonore : {vol_display} → Correct. "
            "La voix est bien projetée, adaptée à un contexte d'entretien."
        )
    else:
        lines.append(
            f"• Volume sonore : {vol_display} → Élevé. "
            "Veiller à ne pas paraître trop agressif ou anxieux."
        )

    # Tempo (WPM — mots par minute calculés depuis transcription + durée de parole)
    if tempo_val < 110:
        lines.append(
            f"• Débit vocal : {round(tempo_val, 1)} mots/min ({label_tempo}) → Trop lent. "
            "Cela peut donner une impression de manque de confiance ou d'énergie."
        )
    elif tempo_val <= 170:
        lines.append(
            f"• Débit vocal : {round(tempo_val, 1)} mots/min ({label_tempo}) → Idéal. "
            "Le rythme favorise la compréhension et maintient l'attention de l'interlocuteur."
        )
    else:
        lines.append(
            f"• Débit vocal : {round(tempo_val, 1)} mots/min ({label_tempo}) → Trop rapide. "
            "Un débit élevé indique souvent du stress et rend le discours difficile à suivre."
        )

    # Pauses
    if pause_ratio_pct < 10:
        lines.append(
            f"• Ratio de pauses : {pause_ratio_pct}% → Insuffisant. "
            "Le discours manque d'aération ; des pauses courtes aident à structurer les idées."
        )
    elif pause_ratio_pct <= 30:
        lines.append(
            f"• Ratio de pauses : {pause_ratio_pct}% → Équilibré. "
            "Les silences structurent bien le discours et facilitent la compréhension."
        )
    else:
        lines.append(
            f"• Ratio de pauses : {pause_ratio_pct}% → Nombreuses pauses. "
            "Elles peuvent indiquer des hésitations fréquentes ou une perte du fil conducteur."
        )

    # Tics de langage
    if filler_count == 0:
        lines.append(
            "• Tics de langage : aucun détecté → Excellent. "
            "Le discours est fluide et parfaitement maîtrisé."
        )
    elif filler_count <= 2:
        lines.append(
            f"• Tics de langage : {filler_count} détecté(s) → Acceptable. "
            "Quelques hésitations légères, sans impact majeur sur la qualité."
        )
    elif filler_count <= 5:
        top_fillers = ", ".join(f'"{k}"' for k in list(fillers_dict.keys())[:3])
        lines.append(
            f"• Tics de langage : {filler_count} détectés ({top_fillers}) → Préoccupant. "
            "Ces hésitations altèrent la fluidité et la qualité perçue du discours."
        )
    else:
        top_fillers = ", ".join(f'"{k}" ({v}x)' for k, v in list(fillers_dict.items())[:4])
        lines.append(
            f"• Tics de langage : {filler_count} détectés ({top_fillers}) → Très problématique. "
            "Ce nombre élevé de mots parasites pénalise fortement la note globale."
        )

    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # 3. ANALYSE DE LA TRANSCRIPTION
    # ══════════════════════════════════════════════════════════════
    lines.append(sep)
    lines.append("3. ANALYSE DE LA TRANSCRIPTION")
    lines.append(sep)

    lines.append(f'Transcription : « {transcription} »')
    lines.append("")

    # Longueur du discours
    if wc < 15:
        lines.append(
            f"Le discours est très court ({wc} mots). "
            "Une réponse développée est attendue en entretien professionnel. "
            "Visez au minimum 40 à 60 mots pour une réponse complète."
        )
    elif wc < 40:
        lines.append(
            f"Le discours est de longueur modérée ({wc} mots). "
            "Les réponses pourraient être davantage développées avec des exemples concrets."
        )
    elif wc < 100:
        lines.append(
            f"Le discours est de longueur correcte ({wc} mots). "
            "La densité de contenu est adaptée à un échange en entretien."
        )
    else:
        lines.append(
            f"Le discours est bien développé ({wc} mots). "
            "Veillez cependant à rester concis et ciblé sur la question posée."
        )

    # Détail des tics
    if filler_count > 0 and fillers_dict:
        filler_detail = ", ".join(
            [f'"{k}" ({v} occurrence{"s" if v > 1 else ""})' for k, v in fillers_dict.items()]
        )
        lines.append(
            f"Tics de langage identifiés : {filler_detail}. "
            "Ces mots parasites fragmentent le discours et donnent une impression d'hésitation. "
            "Il est recommandé de les remplacer par un silence bref et conscient."
        )
    else:
        lines.append(
            "Aucun tic de langage relevé — excellent contrôle de l'expression orale."
        )

    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # 4. FEEDBACK PERSONNALISÉ
    # ══════════════════════════════════════════════════════════════
    lines.append(sep)
    lines.append("4. FEEDBACK PERSONNALISÉ")
    lines.append(sep)

    # ── Points forts ──────────────────────────────────────────────
    points_forts: list[str] = []

    if 30 <= vol_pct <= 70:
        points_forts.append(
            f"Volume vocal adapté ({vol_display}) — la voix est bien projetée et agréable à entendre."
        )
    if 90 <= tempo_val <= 130:
        points_forts.append(
            f"Débit vocal idéal ({round(tempo_val, 1)} BPM) — le rythme est agréable à l'écoute et facilite la compréhension."
        )
    if 10 <= pause_ratio_pct <= 30:
        points_forts.append(
            f"Gestion des silences équilibrée ({pause_ratio_pct}%) — les pauses structurent bien le propos."
        )
    if filler_count == 0:
        points_forts.append("Aucun tic de langage — discours fluide et parfaitement maîtrisé.")
    elif filler_count <= 2:
        points_forts.append("Peu de tics de langage — expression orale globalement propre.")
    if "positif" in sentiment_clean.lower():
        points_forts.append("Sentiment positif détecté — attitude constructive et engagée.")
    if score >= 65:
        points_forts.append(f"Score global solide ({score}/100) — performance au-dessus de la moyenne.")
    if not points_forts:
        points_forts.append(
            "Aucun point fort significatif identifié à ce stade — l'ensemble des dimensions est à travailler."
        )

    # ── Axes d'amélioration ───────────────────────────────────────
    axes: list[str] = []

    if filler_count > 3:
        mots = ", ".join(f'"{k}"' for k in fillers_dict.keys())
        axes.append(
            f"Éliminer les tics de langage ({filler_count} occurrences : {mots}) — "
            "ces mots parasites nuisent fortement à la crédibilité et pénalisent le score."
        )
    if tempo_val > 130:
        axes.append(
            f"Ralentir le débit ({round(tempo_val, 1)} BPM) — "
            "parler trop vite signale du stress et rend le contenu difficile à suivre."
        )
    elif tempo_val < 90:
        axes.append(
            f"Dynamiser le débit ({round(tempo_val, 1)} BPM) — "
            "un rythme plus soutenu témoigne de davantage d'énergie et de confiance."
        )
    if vol_pct < 30:
        axes.append(
            f"Renforcer la projection vocale ({vol_display}) — "
            "une voix plus assurée renforce la crédibilité et l'assertivité."
        )
    if pause_ratio_pct > 30:
        axes.append(
            f"Réduire les pauses excessives ({pause_ratio_pct}%) — "
            "de trop nombreuses interruptions fragmentent le discours et signalent des hésitations."
        )
    if any(k in emotion_lower for k in ("stress", "anxieux", "nerveux")):
        axes.append(
            "Travailler la gestion du stress — l'émotion 'Stressé' est perceptible dans la voix ; "
            "des techniques de préparation mentale sont recommandées avant l'entretien."
        )
    if wc < 30:
        axes.append(
            "Développer davantage les réponses — un discours plus étoffé, appuyé par des exemples "
            "concrets, témoigne d'une meilleure préparation et d'un plus grand engagement."
        )
    if score < 45:
        axes.append(
            "Structurer le discours — utiliser la méthode STAR (Situation, Tâche, Action, Résultat) "
            "pour construire des réponses claires, concrètes et mémorables."
        )
    if not axes:
        axes.append(
            "Performance globalement satisfaisante — maintenir ce niveau et continuer à pratiquer "
            "régulièrement pour consolider les acquis."
        )

    lines.append("✅ POINTS FORTS :")
    for point in points_forts:
        lines.append(f"  • {point}")
    lines.append("")
    lines.append("🔧 AXES D'AMÉLIORATION :")
    for axe in axes:
        lines.append(f"  • {axe}")
    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # 5. CONSEILS PRATIQUES
    # ══════════════════════════════════════════════════════════════
    lines.append(sep)
    lines.append("5. CONSEILS PRATIQUES")
    lines.append(sep)

    conseils: list[str] = []

    if filler_count > 3:
        conseils.append(
            "Pour éliminer les tics de langage : enregistrez-vous régulièrement et écoutez vos "
            "hésitations. Pratiquez l'exercice du 'silence conscient' — lorsque vous ressentez "
            "le besoin de dire 'euh' ou 'bah', faites une pause silencieuse à la place. "
            "Répétez l'exercice jusqu'à ce que le réflexe soit ancré."
        )

    if tempo_val > 130:
        conseils.append(
            "Pour ralentir le débit : respirez profondément avant de prendre la parole. "
            "Imaginez que vous expliquez quelque chose à quelqu'un qui prend des notes — "
            "ce pace est idéal pour un entretien. Comptez mentalement deux secondes entre "
            "chaque idée principale."
        )
    elif tempo_val < 90:
        conseils.append(
            "Pour dynamiser le débit : lisez des textes à voix haute en cherchant à transmettre "
            "de l'énergie et de l'enthousiasme. Variez l'intonation et accentuez les mots clés "
            "pour rendre le discours plus vivant."
        )

    if any(k in emotion_lower for k in ("stress", "anxieux", "nerveux")):
        conseils.append(
            "Pour gérer le stress avant l'entretien : pratiquez la cohérence cardiaque "
            "(3 à 5 minutes de respiration rythmée : 5 secondes inspiration, 5 secondes expiration). "
            "Préparez vos réponses en amont avec la méthode STAR et simulez des entretiens "
            "avec un ami ou devant un miroir."
        )

    if score < 45 or wc < 30:
        conseils.append(
            "Pour améliorer la structure et la richesse des réponses : adoptez la méthode STAR "
            "(Situation → Tâche → Action → Résultat). Préparez 5 à 10 exemples concrets tirés "
            "de vos expériences et entraînez-vous à les restituer en 90 secondes maximum."
        )

    if vol_pct < 30:
        conseils.append(
            "Pour renforcer la projection vocale : pratiquez des exercices debout, le dos droit, "
            "en projetant la voix comme si vous parliez à quelqu'un à 5 m de vous. "
            "Des cours de prise de parole en public ou de théâtre peuvent être particulièrement bénéfiques."
        )

    if not conseils:
        conseils.append(
            "Continuez à vous entraîner régulièrement avec des simulations d'entretien. "
            "Enregistrez-vous, écoutez-vous, et identifiez vos axes de progression au fil du temps. "
            "Des entretiens blancs avec un interlocuteur réel restent le meilleur exercice de préparation."
        )

    for i, conseil in enumerate(conseils, 1):
        lines.append(f"Conseil {i} :")
        lines.append(f"  {conseil}")
        lines.append("")

    # ══════════════════════════════════════════════════════════════
    lines.append(sep)
    lines.append(f"Score final : {score}/100  |  Interprétation : {interpretation}")
    lines.append(sep)

    # ════════════════════════════════════════════════════════════
    # 6. CONCLUSION LLM
    # ════════════════════════════════════════════════════════════
    if llm_conclusion:
        lines.append(sep)
        lines.append("6. CONCLUSION PERSONNALISÉE")
        lines.append(sep)
        lines.append(llm_conclusion)
        lines.append("")

    return "\n".join(lines), points_forts, axes
