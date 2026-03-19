"""
Client HTTP léger pour Ollama (stdlib uniquement — pas de dépendance externe).

Rôle : générer la conclusion narrative du rapport d'entretien via un LLM local,
       générer des questions d'entretien personnalisées,
       et analyser la pertinence question/réponse.
L'URL du service Ollama est lue depuis la variable d'environnement OLLAMA_BASE_URL
(par défaut : http://ollama:11434, nom du service Docker Compose).
"""

import json
import logging
import os
import urllib.request
import urllib.error
from collections.abc import Generator

logger = logging.getLogger(__name__)

_DEFAULT_URL = "http://ollama:11434"
_DEFAULT_MODEL = "llama3.2:3b"
_TIMEOUT_S = 60  # Ollama sur CPU peut être lent ; 60 s est raisonnable pour 3B params


def _build_prompt(
    score: float,
    interpretation: str,
    emotion: str,
    sentiment_label: str,
    filler_count: int,
    tempo_val: float,
    pause_ratio_pct: float,
    word_count: int,
    points_forts: list[str],
    axes: list[str],
) -> str:
    """Construit le prompt utilisateur à partir des métriques déjà calculées."""
    pf_text = "\n".join(f"- {p}" for p in points_forts) if points_forts else "- (aucun)"
    ax_text = "\n".join(f"- {a}" for a in axes) if axes else "- (aucun)"

    return (
        f"Voici les résultats d'analyse d'un entretien professionnel :\n\n"
        f"• Score final : {score}/100 ({interpretation})\n"
        f"• Émotion dominante : {emotion}\n"
        f"• Sentiment global : {sentiment_label}\n"
        f"• Tics de langage : {filler_count}\n"
        f"• Débit vocal : {round(tempo_val, 1)} BPM\n"
        f"• Ratio de pauses : {pause_ratio_pct}%\n"
        f"• Nombre de mots : {word_count}\n\n"
        f"Points forts identifiés :\n{pf_text}\n\n"
        f"Axes d'amélioration :\n{ax_text}\n\n"
        "En te basant exclusivement sur ces données, rédige en 4 à 6 phrases une conclusion "
        "personnalisée, bienveillante et professionnelle à destination du candidat. "
        "Commence directement par la conclusion, sans phrase d'introduction ni titre. "
        "Style : coach RH expert, ton encourageant mais objectif, en français soutenu."
    )


def generate_conclusion(
    score: float,
    interpretation: str,
    emotion: str,
    sentiment_label: str,
    filler_count: int,
    tempo_val: float,
    pause_ratio_pct: float,
    word_count: int,
    points_forts: list[str],
    axes: list[str],
    model: str | None = None,
    base_url: str | None = None,
) -> str | None:
    """
    Appelle Ollama pour générer une conclusion narrative du rapport.

    Retourne le texte généré (str) si Ollama répond dans le délai imparti,
    ou None en cas d'échec (timeout, service indisponible, modèle manquant).
    En cas d'échec, l'appelant doit simplement omettre la section LLM.

    Args:
        model    : Nom du modèle Ollama (défaut : llama3.2:3b)
        base_url : URL du service Ollama (défaut : OLLAMA_BASE_URL ou http://ollama:11434)
    """
    url = (base_url or os.getenv("OLLAMA_BASE_URL", _DEFAULT_URL)).rstrip("/")
    mdl = model or os.getenv("OLLAMA_MODEL", _DEFAULT_MODEL)
    endpoint = f"{url}/api/generate"

    prompt = _build_prompt(
        score=score,
        interpretation=interpretation,
        emotion=emotion,
        sentiment_label=sentiment_label,
        filler_count=filler_count,
        tempo_val=tempo_val,
        pause_ratio_pct=pause_ratio_pct,
        word_count=word_count,
        points_forts=points_forts,
        axes=axes,
    )

    payload = json.dumps({
        "model": mdl,
        "prompt": prompt,
        "system": (
            "Tu es un coach RH expert en entretien professionnel. "
            "Tu rédiges des conclusions courtes, bienveillantes, précises et en français soutenu. "
            "Tu te bases uniquement sur les données fournies. "
            "Tu n'inventes aucune information."
        ),
        "stream": False,
        "options": {
            "temperature": 0.5,   # Suffisamment créatif sans hallucinations
            "top_p": 0.9,
            "num_predict": 250,   # ~4-6 phrases
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            text = body.get("response", "").strip()
            if not text:
                logger.warning("[OllamaClient] Réponse vide reçue — section LLM omise.")
                return None
            logger.info(f"[OllamaClient] Conclusion générée ({len(text)} caractères).")
            return text

    except urllib.error.URLError as e:
        logger.warning(f"[OllamaClient] Service indisponible ({e}) — section LLM omise.")
        return None
    except TimeoutError:
        logger.warning(f"[OllamaClient] Timeout ({_TIMEOUT_S}s) — section LLM omise.")
        return None
    except Exception as e:
        logger.warning(f"[OllamaClient] Erreur inattendue : {e} — section LLM omise.")
        return None


def generate_conclusion_stream(
    score: float,
    interpretation: str,
    emotion: str,
    sentiment_label: str,
    filler_count: int,
    tempo_val: float,
    pause_ratio_pct: float,
    word_count: int,
    points_forts: list[str],
    axes: list[str],
    model: str | None = None,
    base_url: str | None = None,
) -> Generator[str, None, None]:
    """
    Version streaming de generate_conclusion.
    Yield les tokens un par un au fur et à mesure qu'Ollama les génère.
    """
    url = (base_url or os.getenv("OLLAMA_BASE_URL", _DEFAULT_URL)).rstrip("/")
    mdl = model or os.getenv("OLLAMA_MODEL", _DEFAULT_MODEL)
    endpoint = f"{url}/api/generate"

    prompt = _build_prompt(
        score=score,
        interpretation=interpretation,
        emotion=emotion,
        sentiment_label=sentiment_label,
        filler_count=filler_count,
        tempo_val=tempo_val,
        pause_ratio_pct=pause_ratio_pct,
        word_count=word_count,
        points_forts=points_forts,
        axes=axes,
    )

    payload = json.dumps({
        "model": mdl,
        "prompt": prompt,
        "system": (
            "Tu es un coach RH expert en entretien professionnel. "
            "Tu rédiges des conclusions courtes, bienveillantes, précises et en français soutenu. "
            "Tu te bases uniquement sur les données fournies. "
            "Tu n'inventes aucune information."
        ),
        "stream": True,
        "options": {
            "temperature": 0.5,
            "top_p": 0.9,
            "num_predict": 250,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done", False):
                    break
    except urllib.error.URLError as e:
        logger.warning(f"[OllamaClient] Service indisponible ({e}) — stream interrompu.")
    except TimeoutError:
        logger.warning(f"[OllamaClient] Timeout ({_TIMEOUT_S}s) — stream interrompu.")
    except Exception as e:
        logger.warning(f"[OllamaClient] Erreur inattendue : {e} — stream interrompu.")


# ═══════════════════════════════════════════════════════════════
# GÉNÉRATION DE QUESTION D'ENTRETIEN
# ═══════════════════════════════════════════════════════════════

def generate_interview_question(
    domain: str,
    position: str,
    focus_points: str,
    model: str | None = None,
    base_url: str | None = None,
) -> str | None:
    """
    Génère une question d'entretien personnalisée à partir du profil candidat.
    Retourne la question (str) ou None si Ollama est indisponible.
    """
    url = (base_url or os.getenv("OLLAMA_BASE_URL", _DEFAULT_URL)).rstrip("/")
    mdl = model or os.getenv("OLLAMA_MODEL", _DEFAULT_MODEL)
    endpoint = f"{url}/api/generate"

    prompt = (
        f"Profil du candidat :\n"
        f"• Domaine de compétences : {domain}\n"
        f"• Type de poste recherché : {position}\n"
        f"• Points à travailler en entretien : {focus_points}\n\n"
        "Génère UNE SEULE question d'entretien professionnel réaliste et pertinente "
        "pour ce profil. La question doit être ouverte, stimulante et permettre au "
        "candidat de démontrer ses compétences comportementales (soft skills). "
        "Donne uniquement la question, sans numéro, sans introduction, sans explication."
    )

    payload = json.dumps({
        "model": mdl,
        "prompt": prompt,
        "system": (
            "Tu es un recruteur RH senior spécialisé en entretiens professionnels. "
            "Tu formules des questions d'entretien précises, réalistes et adaptées au profil du candidat. "
            "Tu ne poses qu'une seule question à la fois. Tu réponds en français."
        ),
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 150,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            text = body.get("response", "").strip()
            if not text:
                logger.warning("[OllamaClient] Réponse vide — question non générée.")
                return None
            logger.info(f"[OllamaClient] Question générée ({len(text)} caractères).")
            return text
    except urllib.error.URLError as e:
        logger.warning(f"[OllamaClient] Service indisponible ({e}) — question non générée.")
        return None
    except TimeoutError:
        logger.warning(f"[OllamaClient] Timeout ({_TIMEOUT_S}s) — question non générée.")
        return None
    except Exception as e:
        logger.warning(f"[OllamaClient] Erreur inattendue : {e} — question non générée.")
        return None


# ═══════════════════════════════════════════════════════════════
# ANALYSE DE PERTINENCE QUESTION/RÉPONSE (STREAMING)
# ═══════════════════════════════════════════════════════════════

def generate_relevance_stream(
    question: str,
    transcription: str,
    domain: str,
    position: str,
    model: str | None = None,
    base_url: str | None = None,
) -> Generator[str, None, None]:
    """
    Analyse la pertinence entre la question posée et la réponse du candidat.
    Yield les tokens au fur et à mesure.
    """
    url = (base_url or os.getenv("OLLAMA_BASE_URL", _DEFAULT_URL)).rstrip("/")
    mdl = model or os.getenv("OLLAMA_MODEL", _DEFAULT_MODEL)
    endpoint = f"{url}/api/generate"

    prompt = (
        f"Contexte :\n"
        f"• Domaine du candidat : {domain}\n"
        f"• Poste recherché : {position}\n\n"
        f"Question posée au candidat :\n« {question} »\n\n"
        f"Réponse du candidat (transcription verbatim) :\n« {transcription} »\n\n"
        "Analyse la pertinence de la réponse par rapport à la question posée. "
        "Évalue en 4 à 6 phrases : "
        "1) Le candidat a-t-il bien compris et répondu à la question ? "
        "2) La réponse est-elle structurée et argumentée ? "
        "3) Les exemples ou arguments sont-ils pertinents pour le poste visé ? "
        "4) Quels conseils concrets donnerais-tu pour améliorer la réponse ? "
        "Commence directement par l'analyse, sans titre ni introduction. "
        "Style : coach RH bienveillant mais objectif, en français soutenu."
    )

    payload = json.dumps({
        "model": mdl,
        "prompt": prompt,
        "system": (
            "Tu es un coach RH expert en préparation aux entretiens professionnels. "
            "Tu analyses la cohérence et la pertinence des réponses de candidats. "
            "Tu es constructif, précis et bienveillant. Tu réponds en français soutenu."
        ),
        "stream": True,
        "options": {
            "temperature": 0.5,
            "top_p": 0.9,
            "num_predict": 300,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done", False):
                    break
    except urllib.error.URLError as e:
        logger.warning(f"[OllamaClient] Service indisponible ({e}) — pertinence non analysée.")
    except TimeoutError:
        logger.warning(f"[OllamaClient] Timeout ({_TIMEOUT_S}s) — pertinence non analysée.")
    except Exception as e:
        logger.warning(f"[OllamaClient] Erreur inattendue : {e} — pertinence non analysée.")
