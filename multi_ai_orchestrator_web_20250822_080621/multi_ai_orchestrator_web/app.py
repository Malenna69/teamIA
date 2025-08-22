# -*- coding: utf-8 -*-
"""
EcoSwitch — Team IA Orchestrator (Streamlit)
Version corrigée (2025-08-22)

Correctifs inclus:
1) Décideur auto: clamp de l'index + fallback heuristique pour éviter IndexError.
2) UI Orchestrateur: appel du décideur protégé par try/except + affichage conditionnel.
3) Mode Chat: sélection correcte du provider du juge (gpt/grok/gemini vs heuristic),
   au lieu de passer "llm" directement.
4) RAG: extraction PDF robuste, build index tolérant, erreurs affichées via st.exception.
"""

import os, json, time, asyncio, re, io, datetime
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

import streamlit as st
import pandas as pd
import numpy as np

from tenacity import retry, stop_after_attempt, wait_exponential

# LLM SDKs
from openai import OpenAI, AsyncOpenAI
from google import genai
from google.genai import types as genai_types

# PDF for RAG
from pypdf import PdfReader

# ========================= Constantes =========================
TIMEOUT_S = 60
MAX_CHARS = 14000           # coupe les sorties pour éviter débordements
RAG_TOPK = 4                # nb de passages injectés
EMBED_MODEL = "text-embedding-3-small"
APP_NAME = "🤝 Team IA – EcoSwitch"

# ========================= Utils =========================
def _get_key(name: str) -> Optional[str]:
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name)

def _get_temp(default_str: str, ui_value: Optional[float]) -> float:
    if ui_value is not None:
        return float(ui_value)
    try:
        return float(os.getenv("MODEL_TEMPERATURE", default_str))
    except Exception:
        return float(default_str)

def _trim(text: str, limit: int = MAX_CHARS) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "\n\n[...trimmed...]"

def _now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@dataclass
class ProviderResult:
    provider: str
    model: str
    output: str
    latency_s: float
    ok: bool
    error: Optional[str] = None

# ========================= Presets “Départements” =========================
DEPARTMENTS: Dict[str, Dict[str, Any]] = {
    "general": {
        "label": "Général / CEO",
        "weights": {"rigor":0.40,"usefulness":0.30,"creativity":0.20,"risk":0.10},
        "system": "Tu es un board exécutif multidisciplinaire d'EcoSwitch. Stratégie claire, priorisation, actions concrètes.",
        "temp": 0.7, "debate": 0
    },
    "marketing": {
        "label": "Marketing B2B",
        "weights": {"rigor":0.30,"usefulness":0.40,"creativity":0.20,"risk":0.10},
        "system": "Tu es le département marketing B2B d'EcoSwitch. Acquisition, contenu persuasif, ROI data-driven.",
        "temp": 0.8, "debate": 1
    },
    "coding": {
        "label": "Dev / Coding",
        "weights": {"rigor":0.50,"usefulness":0.30,"creativity":0.10,"risk":0.10},
        "system": "Tu es le département dev full-stack d'EcoSwitch. Code Python/JS propre, testable, prêt prod.",
        "temp": 0.4, "debate": 0
    },
    "ingenierie": {
        "label": "Ingénierie / IoT",
        "weights": {"rigor":0.45,"usefulness":0.35,"creativity":0.10,"risk":0.10},
        "system": "Tu es le département ingénierie d'EcoSwitch. IoT, systèmes, simulations, trade-offs techniques.",
        "temp": 0.6, "debate": 2
    },
    "thermie-batiment": {
        "label": "Thermie du bâtiment",
        "weights": {"rigor":0.50,"usefulness":0.25,"creativity":0.15,"risk":0.10},
        "system": "Tu es le département thermique. Normes RE2020, HVAC, métriques énergétiques, calculs.",
        "temp": 0.5, "debate": 1
    },
    "product": {
        "label": "Product Management",
        "weights": {"rigor":0.35,"usefulness":0.40,"creativity":0.15,"risk":0.10},
        "system": "Tu es le département product. Roadmap, user stories, MVP, priorisation RICE.",
        "temp": 0.7, "debate": 1
    },
    "ux-ui": {
        "label": "UX / UI",
        "weights": {"rigor":0.30,"usefulness":0.30,"creativity":0.30,"risk":0.10},
        "system": "Tu es le département UX/UI. Wireframes textuels, user journeys, accessibilité.",
        "temp": 0.8, "debate": 1
    },
    "finance": {
        "label": "Finance & Pricing",
        "weights": {"rigor":0.50,"usefulness":0.30,"creativity":0.10,"risk":0.10},
        "system": "Tu es le département finance. Pricing SaaS, ROI, cohortes, projections.",
        "temp": 0.5, "debate": 0
    },
    "legal": {
        "label": "Legal & Compliance",
        "weights": {"rigor":0.60,"usefulness":0.25,"creativity":0.05,"risk":0.10},
        "system": "Tu es le département legal. GDPR, ESG, contrats B2B, checklists, risques.",
        "temp": 0.3, "debate": 1
    },
    "operations": {
        "label": "Ops & Scaling",
        "weights": {"rigor":0.45,"usefulness":0.35,"creativity":0.10,"risk":0.10},
        "system": "Tu es le département ops. Déploiement cloud, sécurité, monitoring, SLO.",
        "temp": 0.6, "debate": 1
    },
    "sustainability": {
        "label": "Sustainability R&D",
        "weights": {"rigor":0.40,"usefulness":0.25,"creativity":0.25,"risk":0.10},
        "system": "Tu es la R&D sustainability. Tendances énergie, IA carbone, prototypes.",
        "temp": 0.7, "debate": 2
    },
    "sales": {
        "label": "Sales & Customer Success",
        "weights": {"rigor":0.30,"usefulness":0.40,"creativity":0.20,"risk":0.10},
        "system": "Tu es le département sales. Pitches, objections, onboarding, KPI.",
        "temp": 0.7, "debate": 1
    },
}

def _apply_department(preset_key: str, system_txt: str, temp_val: float, debate_val: int):
    cfg = DEPARTMENTS.get(preset_key, {})
    weights = cfg.get("weights", {"rigor":0.40,"usefulness":0.30,"creativity":0.20,"risk":0.10})
    system = cfg.get("system", system_txt)
    temp = cfg.get("temp", temp_val)
    debate = cfg.get("debate", debate_val)
    return weights, system, temp, debate

# ========================= Providers (sync + retries) =========================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
def ask_openai_gpt(prompt: str, system: Optional[str], model: str, temp: Optional[float]) -> ProviderResult:
    api_key = _get_key("OPENAI_API_KEY")
    if not api_key:
        return ProviderResult("gpt", model, "", 0.0, False, "OPENAI_API_KEY manquant")
    client = OpenAI(api_key=api_key, timeout=TIMEOUT_S)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content": system or "You are a rigorous, neutral assistant."},
                {"role":"user","content": prompt}
            ],
            temperature=_get_temp("0.7", temp)
        )
        txt = _trim(resp.choices[0].message.content or "")
        return ProviderResult("gpt", model, txt, time.time()-t0, True)
    except Exception as e:
        return ProviderResult("gpt", model, "", time.time()-t0, False, str(e))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
def ask_grok_xai(prompt: str, system: Optional[str], model: str, temp: Optional[float]) -> ProviderResult:
    api_key = _get_key("XAI_API_KEY")
    if not api_key:
        return ProviderResult("grok", model, "", 0.0, False, "XAI_API_KEY manquant")
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1", timeout=TIMEOUT_S)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content": system or "Be precise, insightful and practical."},
                {"role":"user","content": prompt}
            ],
            temperature=_get_temp("0.8", temp)
        )
        txt = _trim(resp.choices[0].message.content or "")
        return ProviderResult("grok", model, txt, time.time()-t0, True)
    except Exception as e:
        return ProviderResult("grok", model, "", time.time()-t0, False, str(e))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
def ask_gemini(prompt: str, system: Optional[str], model: str, temp: Optional[float]) -> ProviderResult:
    api_key = _get_key("GOOGLE_API_KEY")
    if not api_key:
        return ProviderResult("gemini", model, "", 0.0, False, "GOOGLE_API_KEY manquant")
    client = genai.Client(api_key=api_key)
    t0 = time.time()
    try:
        cfg = genai_types.GenerateContentConfig(
            system_instruction=system or "You are a clear, structured, neutral expert.",
            temperature=_get_temp("0.6", temp),
        )
        res = client.models.generate_content(
            model=model,
            contents=prompt,
            config=cfg
        )
        txt = _trim((res.text or ""))
        return ProviderResult("gemini", model, txt, time.time()-t0, True)
    except Exception as e:
        return ProviderResult("gemini", model, "", time.time()-t0, False, str(e))

# ========================= Providers (async) =========================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
async def ask_openai_gpt_async(prompt: str, system: Optional[str], model: str, temp: Optional[float]) -> ProviderResult:
    api_key = _get_key("OPENAI_API_KEY")
    if not api_key:
        return ProviderResult("gpt", model, "", 0.0, False, "OPENAI_API_KEY manquant")
    client = AsyncOpenAI(api_key=api_key, timeout=TIMEOUT_S)
    t0 = time.time()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content": system or "You are a rigorous, neutral assistant."},
                {"role":"user","content": prompt}
            ],
            temperature=_get_temp("0.7", temp)
        )
        txt = _trim(resp.choices[0].message.content or "")
        return ProviderResult("gpt", model, txt, time.time()-t0, True)
    except Exception as e:
        return ProviderResult("gpt", model, "", time.time()-t0, False, str(e))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
async def ask_grok_xai_async(prompt: str, system: Optional[str], model: str, temp: Optional[float]) -> ProviderResult:
    api_key = _get_key("XAI_API_KEY")
    if not api_key:
        return ProviderResult("grok", model, "", 0.0, False, "XAI_API_KEY manquant")
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.x.ai/v1", timeout=TIMEOUT_S)
    t0 = time.time()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content": system or "Be precise, insightful and practical."},
                {"role":"user","content": prompt}
            ],
            temperature=_get_temp("0.8", temp)
        )
        txt = _trim(resp.choices[0].message.content or "")
        return ProviderResult("grok", model, txt, time.time()-t0, True)
    except Exception as e:
        return ProviderResult("grok", model, "", time.time()-t0, False, str(e))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
async def ask_gemini_async(prompt: str, system: Optional[str], model: str, temp: Optional[float]) -> ProviderResult:
    return await asyncio.to_thread(ask_gemini, prompt, system, model, temp)

# ========================= Heuristic scoring =========================
def heuristic_scores(text: str):
    t = text or ""
    words = re.findall(r"\w+", t.lower())
    unique = len(set(words)) if words else 1
    rigor = 2.0*len(re.findall(r"\b(en|iso|din|norme|source|ref|réf)\b", t.lower())) \
          + 1.0*len(re.findall(r"\d+(\.\d+)?", t)) \
          + 1.5*len(re.findall(r"^\d+[\.\)]", t, flags=re.M))
    usefulness = 1.5*len(re.findall(r"\b(\- |\* )", t)) \
               + 2.0*len(re.findall(r"\b(étape|step|plan|roadmap|livrable|deadline)\b", t.lower())) \
               + 1.0*len(re.findall(r"\b(implémente|construis|déploie|mesure|teste|valide)\b", t.lower()))
    creativity = min(10.0, unique / max(1, len(words)) * 20.0)
    risk = 2.0*len(re.findall(r"\b(garanti|100%|parfait|jamais|toujours)\b", t.lower())) \
         + 1.0*len(re.findall(r"\b(magique|révolution)\b", t.lower()))
    clamp = lambda x: max(0.0, min(10.0, x))
    return {"rigor": clamp(rigor/5.0), "usefulness": clamp(usefulness/5.0),
            "creativity": clamp(creativity), "risk": clamp(risk)}

def _total_score(s: Dict[str,float], w: Dict[str,float]) -> float:
    return w["rigor"]*s["rigor"] + w["usefulness"]*s["usefulness"] + w["creativity"]*s["creativity"] + w["risk"]*(10 - s["risk"])

# ========================= Judge =========================
def judge_with_provider(task: str, entries: list, provider: str, judge_model: Optional[str], weights: Dict[str,float]):
    if provider == "heuristic":
        scores = []
        for e in entries:
            s = heuristic_scores(e["output"]); s["provider"] = e["provider"]; scores.append(s)
        totals = [(s["provider"], _total_score(s, weights)) for s in scores]
        ranking = [p for p,_ in sorted(totals, key=lambda x: x[1], reverse=True)]
        return {"scores": scores, "weighted_ranking": ranking,
                "final_synthesis": "Synthèse non-LLM: utiliser le top-rank comme base et fusionner.",
                "action_plan": "- Prendre les points clés du meilleur score\n- Ajouter 2 actions du 2e meilleur\n- Lister les risques signalés"}

    bundle = {"task": task, "entries": entries, "weights": weights}
    prompt = (
        "You are an impartial evaluator.\n"
        "Return STRICT JSON only with keys: scores[], weighted_ranking[], final_synthesis, action_plan.\n"
        "Scores items have fields: provider, rigor, usefulness, creativity, risk (0..10).\n"
        "Compute weighted ranking using weights; lower risk increases total via (10-risk).\n\n"
        f"DATA:\n{json.dumps(bundle, ensure_ascii=False)}"
    )

    if provider in ("gpt","grok"):
        base_url = "https://api.x.ai/v1" if provider=="grok" else None
        api_key = _get_key("XAI_API_KEY") if provider=="grok" else _get_key("OPENAI_API_KEY")
        if not api_key: raise RuntimeError("Clé API manquante pour le juge sélectionné.")
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=TIMEOUT_S)
        resp = client.chat.completions.create(
            model=judge_model or ("grok-4" if provider=="grok" else "gpt-4o-mini"),
            messages=[
                {"role":"system","content":"Return ONLY JSON per instructions. No preface."},
                {"role":"user","content":prompt}
            ],
            temperature=0.0,
            response_format={"type":"json_object"} if provider=="gpt" else None
        )
        txt = resp.choices[0].message.content
        try:
            return json.loads(txt)
        except Exception:
            s = txt.find("{"); e = txt.rfind("}")
            return json.loads(txt[s:e+1])
    elif provider == "gemini":
        api_key = _get_key("GOOGLE_API_KEY")
        if not api_key: raise RuntimeError("GOOGLE_API_KEY manquante pour juge Gemini.")
        client = genai.Client(api_key=api_key)
        cfg = genai_types.GenerateContentConfig(
            system_instruction="Return ONLY valid minified JSON. Do not include markdown.",
            temperature=0.0,
        )
        res = client.models.generate_content(
            model=judge_model or "gemini-2.5-pro",
            contents=prompt,
            config=cfg
        )
        return json.loads(res.text)
    else:
        raise RuntimeError("Juge inconnu.")

# ========================= Debate mode =========================
def role_system(base: Optional[str]) -> str:
    base = base or "You are a respectful, concise expert."
    extra = (
        " You are in a multi-agent debate. "
        "Each round, first write a brief CRITIQUE (<=120 words) of others' drafts, "
        "then produce an improved REVISION. "
        "Format:\nCRITIQUE: ...\nREVISION:\n..."
    )
    return base + extra

def prompt_for_round(task: str, self_name: str, drafts: Dict[str, str], round_idx: int) -> str:
    others = {k:v for k,v in drafts.items() if k != self_name}
    def _t(txt: str, limit=4000): return txt[:limit]
    blocks = [f"Task:\n{task}\n", f"Round: {round_idx}\n"]
    for name, draft in others.items():
        blocks.append(f"=== {name.upper()} CURRENT DRAFT ===\n{_t(draft)}\n")
    blocks.append(
        "Instructions:\n"
        "- Provide CRITIQUE (succinct, concrete) on gaps, errors, structure.\n"
        "- Then provide a clear, improved REVISION (self-contained)."
    )
    return "\n".join(blocks)

async def run_debate_async(task: str, system: Optional[str], rounds: int,
                           use_gpt: bool, use_grok: bool, use_gemini: bool,
                           gpt_model: str, grok_model: str, gemini_model: str,
                           temp: Optional[float], progress=None):
    participants: List[Tuple[str, Any, str]] = []
    if use_gpt: participants.append(("gpt", ask_openai_gpt_async, gpt_model))
    if use_grok: participants.append(("grok", ask_grok_xai_async, grok_model))
    if use_gemini: participants.append(("gemini", ask_gemini_async, gemini_model))

    drafts: Dict[str,str] = {}
    transcript: List[Dict[str,Any]] = []

    async def _initial(name, fn, model):
        r = await fn(task, role_system(system), model, temp)
        text = r.output if r.ok else f"[ERROR] {r.error}"
        return name, model, text

    init_results = await asyncio.gather(*[_initial(n,f,m) for (n,f,m) in participants])
    for name, model, text in init_results:
        drafts[name] = text
        transcript.append({"round": 0, "speaker": name, "model": model, "text": text})

    for rd in range(1, rounds+1):
        async def _round_call(name, fn, model, prompt):
            r = await fn(prompt, role_system(system), model, temp)
            text = r.output if r.ok else f"[ERROR] {r.error}"
            rev = text
            up = text.upper()
            if "REVISION:" in up:
                idx = up.find("REVISION:")
                rev = text[idx+9:].strip()
            return name, model, text, (rev or text)

        prompts = [(name, fn, model, prompt_for_round(task, name, drafts, rd)) for (name, fn, model) in participants]
        round_results = await asyncio.gather(*[ _round_call(name, fn, model, p) for (name, fn, model, p) in prompts ])
        new_drafts = {}
        for name, model, text, rev in round_results:
            transcript.append({"round": rd, "speaker": name, "model": model, "text": text})
            new_drafts[name] = rev
        drafts = new_drafts

    return drafts, transcript

# ========================= RAG (KB) =========================
def _chunk_text(txt: str, size: int = 1200, overlap: int = 200) -> List[str]:
    if not txt: return []
    chunks = []
    start = 0
    n = len(txt)
    while start < n:
        end = min(n, start + size)
        chunk = txt[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0: start = 0
        if start >= n: break
    return chunks

# PATCH: extraction PDF robuste

def _pdf_to_text(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        try:
            if getattr(reader, "is_encrypted", False):
                try:
                    reader.decrypt("")
                except Exception:
                    return ""
        except Exception:
            return ""
        out = []
        for page in reader.pages:
            try:
                out.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(out).strip()
    except Exception:
        return ""


def _ext_to_text(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return _pdf_to_text(data)
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None: return -1.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)

# PATCH: build KB tolérant

def build_kb_from_files(files: List[Any]) -> List[Dict[str, Any]]:
    texts = []
    for f in files or []:
        try:
            data = f.getvalue()
            txt = _ext_to_text(f.name, data)
        except Exception:
            txt = ""
        if not (txt or "").strip():
            continue
        for ch in _chunk_text(txt):
            if ch.strip():
                texts.append({"source": f.name, "text": ch})
    return texts


def embed_texts(texts: List[str]) -> List[List[float]]:
    api_key = _get_key("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("RAG: OPENAI_API_KEY requis pour embeddings.")
    client = OpenAI(api_key=api_key, timeout=TIMEOUT_S)
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def search_kb(q: str, kb: List[Dict[str,Any]], topk: int = RAG_TOPK) -> str:
    if not kb: return ""
    q_emb = embed_texts([q])[0]
    sims = []
    for itm in kb:
        emb = np.array(itm["embedding"], dtype=np.float32)
        sims.append((_cosine(np.array(q_emb, dtype=np.float32), emb), itm))
    sims.sort(key=lambda x: x[0], reverse=True)
    picked = [it["text"] for _, it in sims[:topk]]
    return "\n\n".join(picked)

# ========================= Décideur auto (10/10 light) =========================
async def decideur_auto(entries: List[Dict], weights: Dict[str, float], user_criteria: str,
                        project_ctx: str = "", max_agents: int = 3, max_loops: int = 2) -> Dict[str, Any]:
    """
    Multi-agents (Tech, Business, Éco) -> JSON (choix, explication, plan, alternatives, risques)
    Boucle de raffinement; fallback heuristique si JSON invalide.
    """
    if not entries:
        return {"error": "Aucun output à analyser."}

    # 1) Construire le contexte
    outputs = [e["output"] for e in entries]
    base_ctx = f"Contexte projet EcoSwitch: {project_ctx}\nCritères: {user_criteria}\nPoids: {weights}\n"
    data_blob = f"Outputs à évaluer:\n" + "\n\n---\n\n".join([_trim(t, 3000) for t in outputs])

    # 2) Agents spécialisés (prompts)
    focuses = ["Technique/Faisabilité", "Business/ROI", "Écologie/ESG"]
    agents = []

    async def run_agent(focus: str):
        prompt = (
            f"{base_ctx}\nRôle: Expert {focus}.\n"
            "Tâches:\n"
            "1) Noter chaque output (0..10) sur rigor/usefulness/creativity/risk/faisabilité/impact_eco/ROI.\n"
            "2) Comparer et proposer 1 choix.\n"
            "3) Expliquer (XAI: forces/faiblesses, arbitrages).\n"
            "4) Donner un plan d'action (5-10 étapes concrètes).\n"
            "5) Proposer 2 alternatives pertinentes.\n"
            "Format STRICT JSON: {"
            "\"scores\": [{\"idx\":int,\"rigor\":float,\"usefulness\":float,\"creativity\":float,\"risk\":float,"
            "\"faisabilite\":float,\"impact_eco\":float,\"roi\":float}],"
            "\"choix_idx\": int,"
            "\"explication\": str,"
            "\"plan\": [str],"
            "\"alternatives\": [str],"
            "\"risks\": {\"global\": float, \"notes\": str}"
            "}"
            f"\n\n{data_blob}"
        )
        # Utilise Grok pour la fermeté JSON, sinon GPT
        res = await ask_grok_xai_async(prompt, "Agent expert: objectif, concis, JSON only.", "grok-4", 0.4)
        if not res.ok or not res.output:
            res = await ask_openai_gpt_async(prompt, "Return only JSON.", "gpt-4o-mini", 0.2)
        return res.output if res.ok else "{}"

    for f in focuses[:max_agents]:
        agents.append(run_agent(f))
    agent_jsons = await asyncio.gather(*agents)

    # 3) Orchestrateur & boucle de raffinement
    def _safe_json(txt: str) -> dict:
        try:
            s = txt.find("{"); e = txt.rfind("}")
            if s != -1 and e != -1: txt = txt[s:e+1]
            return json.loads(txt)
        except Exception:
            return {}

    agent_parsed = [_safe_json(t) for t in agent_jsons if t]
    synth: Dict[str, Any] = {}
    for loop in range(max_loops):
        prompt = (
            "Tu es l'orchestrateur final. Fusionne les avis agents en une décision claire.\n"
            "Retourne STRICT JSON: {"
            "\"choix_idx\": int, "
            "\"explication\": str, "
            "\"plan\": [str], "
            "\"alternatives\": [str], "
            "\"risks\": {\"global\": float, \"notes\": str}"
            "}\n"
            f"Agents JSON:\n{json.dumps(agent_parsed, ensure_ascii=False)}"
        )
        final = await ask_openai_gpt_async(prompt, "Return JSON only.", "gpt-4o-mini", 0.0)
        synth = _safe_json(final.output if final.ok else "{}")
        if synth.get("choix_idx") is not None:
            break

    # 4) Fallback heuristique si besoin (aucun choix proposé)
    if synth.get("choix_idx") is None:
        scores = [heuristic_scores(e["output"]) for e in entries]
        totals = [(_total_score(s, weights), i) for i,s in enumerate(scores)]
        totals.sort(reverse=True)
        best_idx = totals[0][1]
        synth = {
            "choix_idx": best_idx,
            "explication": "Fallback heuristique (pondéré).",
            "plan": ["Valider les hypothèses clés.", "Prototyper la solution retenue.", "Mesurer l'impact (énergie/ESG)."],
            "alternatives": [f"Option {i}" for i in range(len(entries)) if i != best_idx],
            "risks": {"global": 0.35, "notes": "Dépend des données d'entrée et intégrations techniques."}
        }

    # 5) Enrichir avec infos choix (avec garde anti-hors-bornes)
    n = len(entries)
    try:
        idx = int(synth.get("choix_idx", 0))
    except Exception:
        idx = 0

    if not (0 <= idx < n):
        # Fallback heuristique si l'index proposé est invalide
        scores = [heuristic_scores(e["output"]) for e in entries]
        totals = [(_total_score(s, weights), i) for i, s in enumerate(scores)]
        totals.sort(reverse=True)
        idx = totals[0][1] if totals else 0
        synth["explication"] = (synth.get("explication","") + " (idx hors bornes → fallback heuristique)").strip()

    synth["choix_idx"] = idx
    synth["choix_provider"] = entries[idx]["provider"]
    synth["choix_model"] = entries[idx]["model"]
    synth["choix_excerpt"] = _trim(entries[idx]["output"], 600)

    return synth

# ========================= Exports (MD / HTML) =========================
def make_markdown_report(prompt: str, system: str, dep_key: str, weights: Dict[str,float],
                         entries: List[Dict[str,Any]], scoreboard: Dict[str,Any],
                         decision: Optional[Dict[str,Any]] = None) -> str:
    ts = _now_str()
    md = []
    md.append(f"# Rapport Orchestrateur Multi-IA — EcoSwitch\n\n_Généré le {ts}_\n")
    md.append("## 🎯 Prompt\n```\n" + prompt + "\n```\n")
    md.append(f"**System prompt :** {system}\n")
    md.append("## ⚙️ Paramètres\n")
    models = ", ".join([f"{e['provider']}={e['model']}" for e in entries]) if entries else "—"
    md.append(f"- Modèles: {models}\n")
    md.append(f"- Département: {DEPARTMENTS[dep_key]['label']}\n")
    md.append(f"- Poids: Rigor={weights['rigor']}  Usefulness={weights['usefulness']}  Creativity={weights['creativity']}  Risk={weights['risk']}\n")
    if scoreboard:
        md.append("\n## 📊 Scores\n")
        md.append("| Provider | Rigor | Useful | Creat | Risk | Total |\n|---|---:|---:|---:|---:|---:|")
        rows = []
        for s in scoreboard.get("scores", []):
            total = round(_total_score(s, weights),2)
            rows.append({"provider":s["provider"], "rigor":s["rigor"], "usefulness":s["usefulness"],
                         "creativity":s["creativity"], "risk":s["risk"], "total":total})
        for r in rows:
            md.append(f"| {r['provider']} | {r['rigor']:.2f} | {r['usefulness']:.2f} | {r['creativity']:.2f} | {r['risk']:.2f} | {r['total']:.2f} |")
        ranking = []
        if rows:
            ranking = [x["provider"] for x in sorted(rows, key=lambda z: z["total"], reverse=True)]
        md.append(f"\n**Classement pondéré :** " + (" > ".join(ranking) if ranking else "—") + "\n")

        md.append("\n## 🧩 Synthèse\n")
        md.append(scoreboard.get("final_synthesis","(n/a)") + "\n")
        md.append("\n## ✅ Plan d’action\n")
        md.append(scoreboard.get("action_plan","(n/a)") + "\n")

    if decision:
        md.append("\n## 🧠 Décideur auto (10/10)\n")
        md.append(f"- **Choix** : {decision.get('choix_provider','?')} ({decision.get('choix_model','?')})\n")
        md.append(f"- **Explication** : {decision.get('explication','')}\n")
        md.append(f"- **Risques (global)** : {decision.get('risks',{}).get('global','?')}\n")
        md.append("\n**Plan d’action proposé**\n")
        for i, step in enumerate(decision.get("plan", []), 1):
            md.append(f"{i}. {step}")
        if decision.get("alternatives"):
            md.append("\n**Alternatives**\n- " + "\n- ".join(decision["alternatives"]))
        md.append("\n\n**Extrait du choix**\n```\n" + decision.get("choix_excerpt","") + "\n```\n")

    if entries:
        md.append("\n---\n## 📦 Réponses par modèle\n")
        for e in entries:
            md.append(f"\n### {e['provider'].upper()} • {e['model']} • {e['latency_s']:.2f}s\n\n```\n{e['output']}\n```\n")

    return "\n".join(md)


def make_html_from_markdown(md: str) -> str:
    # Conversion simple (sans lib) : on wrappe le MD dans <pre> pour un rendu quick & propre
    # Pour un vrai rendu MD→HTML, ajoutez markdown2/markdown, mais on reste sans dépendance ici.
    safe = md.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    return f"""<!doctype html>
<html lang=\"fr\"><head>
<meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>Rapport EcoSwitch</title>
<style>
body{{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 32px;}}
pre{{white-space: pre-wrap; word-wrap: break-word; background:#f6f8fa; padding:16px; border-radius:8px;}}
code{{white-space: pre-wrap;}}
h1,h2,h3{{margin-top:1.2em}}
hr{{margin:1.2em 0}}
</style>
</head><body>
<pre>{safe}</pre>
</body></html>"""

# ========================= UI =========================
st.set_page_config(page_title=f"{APP_NAME}", layout="wide")
st.title(APP_NAME)
st.caption("GPT + Grok + Gemini • presets département • async • débat multi-tours • juge pondéré • RAG • Chat")

with st.sidebar:
    st.header("⚙️ Configuration")

    # Mode (Orchestrateur vs Chat)
    mode = st.radio("Mode", ["Orchestrateur", "Chat"], horizontal=True, key="mode_sel")

    # API keys
    st.subheader("Clés API")
    openai_key = st.text_input("OPENAI_API_KEY", value=_get_key("OPENAI_API_KEY") or "", type="password", key="k_openai")
    xai_key    = st.text_input("XAI_API_KEY",    value=_get_key("XAI_API_KEY") or "", type="password", key="k_xai")
    google_key = st.text_input("GOOGLE_API_KEY", value=_get_key("GOOGLE_API_KEY") or "", type="password", key="k_google")
    if openai_key: os.environ["OPENAI_API_KEY"] = openai_key
    if xai_key:    os.environ["XAI_API_KEY"]    = xai_key
    if google_key: os.environ["GOOGLE_API_KEY"] = google_key

    st.subheader("Preset (Département)")
    dep_options = {k:v["label"] for k,v in DEPARTMENTS.items()}
    dep_key = st.selectbox("Sélection", options=list(dep_options.keys()), format_func=lambda k: dep_options[k], index=0, key="dep_sel")

    st.subheader("Fournisseurs actifs")
    colp1, colp2, colp3 = st.columns(3)
    with colp1: use_gpt    = st.checkbox("GPT",    value=True, key="prov_gpt")
    with colp2: use_grok   = st.checkbox("Grok",   value=True, key="prov_grok")
    with colp3: use_gemini = st.checkbox("Gemini", value=True, key="prov_gemini")

    st.subheader("Modèles")
    gpt_model    = st.text_input("GPT model",    "gpt-4o-mini",      key="m_gpt")
    grok_model   = st.text_input("Grok model",   "grok-4",           key="m_grok")
    gemini_model = st.text_input("Gemini model", "gemini-2.5-flash", key="m_gem")

    st.subheader("Paramètres")
    temp          = st.slider("Température", 0.0, 1.5, 0.6, 0.1, key="p_temp")
    debate_rounds = st.number_input("Débat — tours", min_value=0, max_value=5, value=0, step=1, key="p_debate")

    st.subheader("Juge")
    judge_kind     = st.selectbox("Type", ["llm","heuristic"], index=0, key="j_kind")
    judge_provider = st.selectbox("Fournisseur (si llm)", ["gpt","grok","gemini"], index=0, key="j_prov")
    judge_model    = st.text_input("Modèle juge (optionnel)", "", key="j_model")

    st.divider()
    st.subheader("📚 Docs (RAG) — optionnel")
    if "kb" not in st.session_state: st.session_state["kb"] = []
    use_rag  = st.checkbox("Activer RAG (injecter contexte)", value=False, key="rag_use")
    uploaded = st.file_uploader("Ajouter des fichiers .pdf / .txt / .md", type=["pdf","txt","md"], accept_multiple_files=True, key="rag_files")
    if st.button("Construire / Mettre à jour l’index", key="rag_build"):
        with st.spinner("Index RAG en construction…"):
            try:
                kb_texts = build_kb_from_files(uploaded or [])
                kb_texts = [k for k in kb_texts if (k.get("text") or "").strip()]
                if not kb_texts:
                    st.warning("Aucun texte exploitable (PDF chiffré/corrompu ? Essayez un .txt).")
                else:
                    embeds = embed_texts([k["text"] for k in kb_texts])
                    for i, emb in enumerate(embeds):
                        kb_texts[i]["embedding"] = emb
                    st.session_state["kb"] = kb_texts
                    st.success(f"Index prêt ({len(kb_texts)} passages).")
            except Exception as e:
                st.exception(e)

# ========================= MODE ORCHESTRATEUR =========================
if mode == "Orchestrateur":
    prompt = st.text_area("🧠 Prompt", height=160, placeholder="Décris ton besoin…", key="ta_prompt")
    system_default = "Tu es un comité d'experts (ingénierie, UX, marché). Style: clair, structuré, actionnable."
    system = st.text_input("🗣️ System prompt (optionnel)", value=system_default, key="ti_system")

    # Applique preset
    weights, system_applied, temp_applied, debate_applied = _apply_department(dep_key, system, temp, debate_rounds)
    system = system_applied
    temp   = temp_applied
    debate_rounds = debate_applied

    # Décideur - critères (toujours visible)
    st.subheader("Décideur")
    st.text_input("🎛️ Critères décideur", st.session_state.get("dec_crit_orch", "Prioriser faisabilité, coût bas et impact éco haut"), key="dec_crit_orch")


    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        run_btn = st.button("🚀 Lancer", use_container_width=True, key="run_btn_orch")
    with col2:
        save_btn = st.checkbox("Sauvegarde (JSON/MD/HTML)", value=True, key="save_json")
    with col3:
        st.metric("Preset", DEPARTMENTS[dep_key]["label"])

    if run_btn:
        if not prompt.strip():
            st.error("Merci d'écrire un prompt.")
            st.stop()

        # RAG contexte
        rag_ctx = ""
        if use_rag and st.session_state.get("kb"):
            try: rag_ctx = search_kb(prompt, st.session_state["kb"], topk=RAG_TOPK)
            except Exception as e: st.warning(f"RAG désactivé (erreur): {e}")

        full_prompt = prompt if not rag_ctx else f"{prompt}\n\n[Contexte documents]\n{_trim(rag_ctx, 3500)}"

        with st.spinner("Appels LLM en parallèle…"):
            async def run_all():
                if debate_rounds and debate_rounds > 0:
                    final_drafts, tr = await run_debate_async(
                        full_prompt, system, debate_rounds,
                        use_gpt, use_grok, use_gemini,
                        gpt_model, grok_model, gemini_model,
                        temp
                    )
                    out_entries = []
                    for provider, draft in final_drafts.items():
                        out_entries.append({
                            "provider": provider,
                            "model": {"gpt": gpt_model, "grok": grok_model, "gemini": gemini_model}[provider],
                            "latency_s": 0.0, "ok": True, "error": None, "output": _trim(draft)
                        })
                    return out_entries, tr
                else:
                    tasks = []
                    if use_gpt:    tasks.append(ask_openai_gpt_async(full_prompt, system, gpt_model, temp))
                    if use_grok:   tasks.append(ask_grok_xai_async(full_prompt, system, grok_model, temp))
                    if use_gemini: tasks.append(ask_gemini_async(full_prompt, system, gemini_model, temp))
                    results = await asyncio.gather(*tasks) if tasks else []
                    oks = [r for r in results if r.ok]
                    out_entries = [{
                        "provider": r.provider, "model": r.model, "latency_s": r.latency_s,
                        "ok": r.ok, "error": r.error, "output": _trim(r.output)
                    } for r in oks]
                    return out_entries, None

            entries, transcript = asyncio.run(run_all())

        if not entries:
            st.error("Aucune réponse valide — vérifie tes clés API et tes modèles.")
            st.stop()

        # Juge
        provider_for_judge = judge_provider if judge_kind == "llm" else "heuristic"
        try:
            scoreboard = judge_with_provider(prompt, entries, provider_for_judge, judge_model or None, weights)
        except Exception as e:
            st.warning(f"Juge LLM indisponible ({e}). Fallback heuristique.")
            scoreboard = judge_with_provider(prompt, entries, "heuristic", None, weights)

        # Décideur auto (protégé)
        criteria = st.session_state.get("dec_crit_orch", "Prioriser faisabilité, coût bas et impact éco haut")
        try:
            decision = asyncio.run(decideur_auto(entries, weights, criteria, project_ctx=_trim(rag_ctx, 1500)))
        except Exception as e:
            st.warning(f"Décideur auto indisponible : {e}")
            decision = None

        st.success("Terminé ! ✅")

        tab_labels = [f"{e['provider'].upper()}" for e in entries] + ["📊 Scores", "🧠 Décideur"] + (["💬 Débat"] if transcript else [])
        tabs = st.tabs(tab_labels)

        for i, e in enumerate(entries):
            with tabs[i]:
                st.caption(f"Modèle: {e['model']} • Latence: {e['latency_s']:.2f}s")
                st.text_area("Sortie", value=e["output"], height=350, key=f"out_{e['provider']}")

        with tabs[len(entries)]:
            rows = []
            for s in scoreboard.get("scores", []):
                total = round(_total_score(s, weights), 2)
                rows.append({"provider":s["provider"], "rigor":s["rigor"], "usefulness":s["usefulness"],
                             "creativity":s["creativity"], "risk":s["risk"], "total":total})
            if rows:
                df = pd.DataFrame(rows).set_index("provider")
                st.dataframe(df, use_container_width=True)
                ranking = sorted(rows, key=lambda r: r["total"], reverse=True)
                st.write("**Classement pondéré :** " + " > ".join([r['provider'] for r in ranking]))
            st.subheader("Synthèse")
            st.write(scoreboard.get("final_synthesis","(n/a)"))
            st.subheader("Plan d'action")
            st.write(scoreboard.get("action_plan","(n/a)"))

        with tabs[len(entries)+1]:
            if not decision:
                st.info("Décideur non disponible pour cette exécution.")
            else:
                st.write(f"**Choix :** {decision.get('choix_provider')} ({decision.get('choix_model')})")
                st.write("**Explication :**", decision.get("explication",""))
                st.write("**Risques (global)** :", decision.get("risks",{}).get("global"))
                if decision.get("plan"):
                    st.write("**Plan proposé**")
                    for i, step in enumerate(decision["plan"], 1):
                        st.write(f"{i}. {step}")
                if decision.get("alternatives"):
                    st.write("**Alternatives**")
                    for alt in decision["alternatives"]:
                        st.write(f"- {alt}")
                with st.expander("Extrait du choix"):
                    st.code(decision.get("choix_excerpt",""))

        if transcript:
            with tabs[-1]:
                st.write("Transcription du débat (extraits)")
                for rd in sorted(set(t["round"] for t in transcript)):
                    st.markdown(f"#### Round {rd}")
                    for t in [x for x in transcript if x["round"] == rd]:
                        with st.expander(f"{t['speaker'].upper()} ({t['model']})"):
                            st.text(t["text"])

        # Exports
        if save_btn:
            bundle = {
                "prompt": prompt,
                "system": system,
                "department": dep_key,
                "weights": weights,
                "entries": entries,
                "scoreboard": scoreboard,
                "decision": decision,
                "rag_used": bool(rag_ctx),
                "timestamp": _now_str()
            }
            if transcript: bundle["transcript"] = transcript

            # JSON
            st.download_button("⬇️ Export JSON",
                               data=json.dumps(bundle, ensure_ascii=False, indent=2),
                               file_name="results.json",
                               mime="application/json",
                               key="dl_json_orch")

            # Markdown & HTML
            md = make_markdown_report(prompt, system, dep_key, weights, entries, scoreboard, decision)
            st.download_button("⬇️ Export Markdown",
                               data=md, file_name="report.md",
                               mime="text/markdown", key="dl_md_orch")
            html = make_html_from_markdown(md)
            st.download_button("⬇️ Export HTML",
                               data=html, file_name="report.html",
                               mime="text/html", key="dl_html_orch")

# ========================= MODE CHAT (historique + RAG continu) =========================
if mode == "Chat":
    # Etat
    if "chat" not in st.session_state: st.session_state["chat"] = []
    if "chat_cfg" not in st.session_state: st.session_state["chat_cfg"] = {}

    st.subheader("💬 Chat multi-LLM (avec RAG)")
    chat_system_default = "Tu es une équipe IA d'EcoSwitch. Réponds de manière concise, structurée, actionnable."
    chat_system = st.text_input("System (chat)", value=chat_system_default, key="chat_sys")
    weights, chat_system, temp, _ = _apply_department(st.session_state.get("dep_sel","general"), chat_system, st.session_state.get("p_temp",0.6), 0)

    # Affiche l'historique
    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Entrée
    user_msg = st.chat_input("Écris un message…")
    if user_msg:
        # Affiche immédiatement le message user
        st.session_state["chat"].append({"role":"user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        # Construit contexte conversation + RAG
        history_txt = []
        for m in st.session_state["chat"][-6:]:  # on prend les 6 derniers
            role = "Utilisateur" if m["role"]=="user" else "Assistant"
            history_txt.append(f"{role}: {m['content']}")
        convo = "\n".join(history_txt)

        rag_ctx = ""
        if st.session_state.get("rag_use") and st.session_state.get("kb"):
            try:
                rag_ctx = search_kb(user_msg, st.session_state["kb"], topk=RAG_TOPK)
            except Exception:
                rag_ctx = ""

        chat_prompt = (
            f"{convo}\n\nMessage actuel:\n{user_msg}\n\n"
            + (f"[Contexte docs]\n{_trim(rag_ctx, 2500)}\n\n" if rag_ctx else "")
            + "Réponds à l'utilisateur en intégrant le contexte si pertinent."
        )

        # Appels providers en parallèle
        async def chat_call():
            tasks = []
            if st.session_state.get("prov_gpt", True):
                tasks.append(ask_openai_gpt_async(chat_prompt, chat_system, st.session_state.get("m_gpt","gpt-4o-mini"), temp))
            if st.session_state.get("prov_grok", True):
                tasks.append(ask_grok_xai_async(chat_prompt, chat_system, st.session_state.get("m_grok","grok-4"), temp))
            if st.session_state.get("prov_gemini", True):
                tasks.append(ask_gemini_async(chat_prompt, chat_system, st.session_state.get("m_gem","gemini-2.5-flash"), temp))
            results = await asyncio.gather(*tasks) if tasks else []
            entries = [{
                "provider": r.provider, "model": r.model, "latency_s": r.latency_s,
                "ok": r.ok, "error": r.error, "output": _trim(r.output)
            } for r in results if r.ok]
            return entries

        entries = asyncio.run(chat_call())
        if not entries:
            bot_reply = "Aucune réponse valide — vérifie les clés API ou les modèles."
        else:
            # Juge et synthèse courte : on choisit le top et on répond avec son texte
            try:
                kind = st.session_state.get("j_kind","llm")
                prov = st.session_state.get("j_prov","gpt") if kind == "llm" else "heuristic"
                scoreboard = judge_with_provider(user_msg, entries, prov, st.session_state.get("j_model","") or None, weights)
            except Exception:
                scoreboard = judge_with_provider(user_msg, entries, "heuristic", None, weights)

            # tri
            rows = []
            for s in scoreboard.get("scores", []):
                total = round(_total_score(s, weights), 2)
                rows.append({"provider":s["provider"], "total": total})
            top_provider = None
            if rows:
                top_provider = sorted(rows, key=lambda r:r["total"], reverse=True)[0]["provider"]
            chosen = next((e for e in entries if e["provider"] == top_provider), entries[0])
            bot_reply = chosen["output"]

            # Option: petite synthèse en 2 lignes (gpt)
            syn_prompt = (
                "Synthétise en 2 phrases maximum la meilleure réponse pour l'utilisateur, "
                "sans préambule, ton concret et actionnable.\n\n" + bot_reply
            )
            syn = ask_openai_gpt(syn_prompt, "Return concise answer.", st.session_state.get("m_gpt","gpt-4o-mini"), 0.2)
            if syn.ok and syn.output:
                bot_reply = syn.output

        # Affiche et ajoute à l'historique
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
            with st.expander("Voir les réponses par modèle"):
                for e in entries or []:
                    st.caption(f"{e['provider'].upper()} • {e['model']} • {e['latency_s']:.2f}s")
                    st.text_area("Sortie", value=e["output"], height=180, key=f"chat_out_{time.time()}_{e['provider']}")

        st.session_state["chat"].append({"role":"assistant", "content": bot_reply})

    # Outils d’export du chat
    if st.session_state["chat"]:
        chat_json = json.dumps(st.session_state["chat"], ensure_ascii=False, indent=2)
        st.download_button("⬇️ Export Chat (JSON)", data=chat_json, file_name="chat_history.json", mime="application/json", key="dl_chat_json")
        # MD simple du chat
        lines = [f"# Chat {APP_NAME}\n_Généré le {_now_str()}_\n"]
        for m in st.session_state["chat"]:
            role = "Utilisateur" if m["role"]=="user" else "Assistant"
            lines.append(f"**{role}:**\n\n{m['content']}\n")
        chat_md = "\n".join(lines)
        st.download_button("⬇️ Export Chat (MD)", data=chat_md, file_name="chat_history.md", mime="text/markdown", key="dl_chat_md")
        st.download_button("⬇️ Export Chat (HTML)", data=make_html_from_markdown(chat_md), file_name="chat_history.html", mime="text/html", key="dl_chat_html")
