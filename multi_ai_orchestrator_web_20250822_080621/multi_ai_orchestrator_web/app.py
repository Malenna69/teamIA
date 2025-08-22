# -*- coding: utf-8 -*-
import os, json, time, asyncio, re, io
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

import streamlit as st
import pandas as pd
import numpy as np

# Retries
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Providers
from openai import OpenAI, AsyncOpenAI
from google import genai
from google.genai import types as genai_types

# PDF for RAG
from pypdf import PdfReader

# ========================= Constantes =========================
TIMEOUT_S = 60
MAX_CHARS = 14000          # coupe les sorties pour éviter débordements
RAG_TOPK = 4               # nb de passages injectés
EMBED_MODEL = "text-embedding-3-small"

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

# ========================= Providers (async + retries) =========================
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
    # google-genai n'est pas async -> exécute dans un thread
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
                "final_synthesis": "Synthèse non-LLM: utiliser le top-rank comme base et fusionner manuellement.",
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
    def trim(txt: str, limit=4000): return txt[:limit]
    blocks = [f"Task:\n{task}\n", f"Round: {round_idx}\n"]
    for name, draft in others.items():
        blocks.append(f"=== {name.upper()} CURRENT DRAFT ===\n{trim(draft)}\n")
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

    # Round 0 (parallèle)
    async def _initial(name, fn, model):
        r = await fn(task, role_system(system), model, temp)
        text = r.output if r.ok else f"[ERROR] {r.error}"
        return name, model, text

    init_tasks = [_initial(name, fn, model) for (name, fn, model) in participants]
    init_results = await asyncio.gather(*init_tasks)
    for name, model, text in init_results:
        drafts[name] = text
        transcript.append({"round": 0, "speaker": name, "model": model, "text": text})

    if progress: progress.progress(15, text="Débat: initialisation…")

    # Debate rounds
    step = 85 / max(1, rounds)
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
        if progress: progress.progress(min(100, int(15 + rd*step)), text=f"Débat: round {rd}/{rounds}")

    return drafts, transcript

# ========================= RAG (optionnel) =========================
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

def _pdf_to_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    out = []
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(out)

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

def build_kb_from_files(files: List[Any]) -> List[Dict[str, Any]]:
    texts = []
    for f in files or []:
        data = f.getvalue()
        txt = _ext_to_text(f.name, data)
        for ch in _chunk_text(txt):
            texts.append({"source": f.name, "text": ch})
    return texts

def embed_texts(texts: List[str]) -> List[List[float]]:
    api_key = _get_key("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("RAG: OPENAI_API_KEY requis pour embeddings.")
    client = OpenAI(api_key=api_key, timeout=TIMEOUT_S)
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def search_kb(prompt: str, kb: List[Dict[str,Any]], topk: int = RAG_TOPK) -> str:
    if not kb: return ""
    q_emb = embed_texts([prompt])[0]
    sims = []
    for itm in kb:
        emb = np.array(itm["embedding"], dtype=np.float32)
        sims.append((_cosine(np.array(q_emb, dtype=np.float32), emb), itm))
    sims.sort(key=lambda x: x[0], reverse=True)
    picked = [it["text"] for _, it in sims[:topk]]
    ctx = "\n\n".join(picked)
    return ctx

# ========================= UI =========================
st.set_page_config(page_title="Team IA – EcoSwitch (Web)", layout="wide")
st.title("🤝 Team IA – EcoSwitch")
st.caption("GPT + Grok + Gemini • presets département • exécution async • débat multi-tours • juge pondéré • RAG optionnel")

with st.sidebar:
    st.header("⚙️ Configuration")

    mode_simple = st.toggle("Mode simple", value=True, help="Masque des options avancées")

    # API keys
    st.subheader("Clés API")
    openai_key = st.text_input("OPENAI_API_KEY", value=_get_key("OPENAI_API_KEY") or "", type="password", key="k_openai")
    xai_key = st.text_input("XAI_API_KEY", value=_get_key("XAI_API_KEY") or "", type="password", key="k_xai")
    google_key = st.text_input("GOOGLE_API_KEY", value=_get_key("GOOGLE_API_KEY") or "", type="password", key="k_google")
    if openai_key: os.environ["OPENAI_API_KEY"] = openai_key
    if xai_key: os.environ["XAI_API_KEY"] = xai_key
    if google_key: os.environ["GOOGLE_API_KEY"] = google_key

    st.subheader("Preset (Département)")
    dep_options = {k:v["label"] for k,v in DEPARTMENTS.items()}
    dep_key = st.selectbox("Sélection", options=list(dep_options.keys()), format_func=lambda k: dep_options[k], index=0, key="dep_sel")

    st.subheader("Fournisseurs actifs")
    colp1, colp2, colp3 = st.columns(3)
    with colp1: use_gpt = st.checkbox("GPT", value=True, key="prov_gpt")
    with colp2: use_grok = st.checkbox("Grok", value=True, key="prov_grok")
    with colp3: use_gemini = st.checkbox("Gemini", value=True, key="prov_gemini")

    st.subheader("Modèles")
    gpt_model = st.text_input("GPT model", "gpt-4o-mini", key="m_gpt")
    grok_model = st.text_input("Grok model", "grok-4", key="m_grok")
    gemini_model = st.text_input("Gemini model", "gemini-2.5-flash", key="m_gem")

    st.subheader("Paramètres")
    temp = st.slider("Température", 0.0, 1.5, 0.6, 0.1, key="p_temp")
    debate_rounds = st.number_input("Débat — tours", min_value=0, max_value=5, value=0, step=1, key="p_debate")

    st.subheader("Juge")
    judge_kind = st.selectbox("Type", ["llm","heuristic"], index=0, key="j_kind")
    judge_provider = st.selectbox("Fournisseur (si llm)", ["gpt","grok","gemini"], index=0, key="j_prov")
    judge_model = st.text_input("Modèle juge (optionnel)", "", key="j_model")

    st.divider()
    st.subheader("📚 Docs (RAG) — optionnel")
    use_rag = st.checkbox("Activer RAG (injecter contexte)", value=False, key="rag_use")
    uploaded = st.file_uploader("Ajouter des fichiers .pdf / .txt / .md", type=["pdf","txt","md"], accept_multiple_files=True, key="rag_files")
    if "kb" not in st.session_state: st.session_state["kb"] = []
    if st.button("Construire / Mettre à jour l’index", key="rag_build"):
        with st.spinner("Index RAG en construction…"):
            kb_texts = build_kb_from_files(uploaded or [])
            if not kb_texts:
                st.warning("Aucun texte extrait.")
            else:
                try:
                    embeds = embed_texts([k["text"] for k in kb_texts])
                    for i, emb in enumerate(embeds):
                        kb_texts[i]["embedding"] = emb
                    st.session_state["kb"] = kb_texts
                    st.success(f"Index prêt ({len(kb_texts)} passages).")
                except Exception as e:
                    st.error(f"RAG: échec embeddings — {e}")

# Prompt & System
prompt = st.text_area("🧠 Prompt", height=160, placeholder="Décris ton besoin…", key="ta_prompt")
system_default = "Tu es un comité d'experts (ingénierie, UX, marché). Style: clair, structuré, actionnable."
system = st.text_input("🗣️ System prompt (optionnel)", value=system_default, key="ti_system")

# Applique preset (simple mode = on priorise les presets)
weights, system_applied, temp_applied, debate_applied = _apply_department(dep_key, system, temp, debate_rounds)
if mode_simple:
    system = system_applied
    temp = temp_applied
    debate_rounds = debate_applied

col1, col2, col3 = st.columns([1,1,1])
with col1:
    run_btn = st.button("🚀 Lancer", use_container_width=True)
with col2:
    save_btn = st.checkbox("Sauvegarde locale (JSON)", value=True, key="save_json")
with col3:
    st.metric("Preset", DEPARTMENTS[dep_key]["label"])

# ========================= RUN =========================
if run_btn:
    if not prompt.strip():
        st.error("Merci d'écrire un prompt.")
        st.stop()

    # RAG: construit contexte si activé
    rag_ctx = ""
    if use_rag and st.session_state.get("kb"):
        try:
            rag_ctx = search_kb(prompt, st.session_state["kb"], topk=RAG_TOPK)
        except Exception as e:
            st.warning(f"RAG désactivé (erreur): {e}")
            rag_ctx = ""

    # Prompt final (avec contexte si dispo)
    full_prompt = prompt if not rag_ctx else f"{prompt}\n\n[Contexte documents]\n{_trim(rag_ctx, 3500)}"

    progress = st.progress(0, text="Initialisation…")
    with st.spinner("Exécution en parallèle…"):
        entries: List[Dict[str,Any]] = []
        transcript = None

        async def run_parallel_one_shot():
            tasks = []
            if use_gpt:
                tasks.append(ask_openai_gpt_async(full_prompt, system, gpt_model, temp))
            if use_grok:
                tasks.append(ask_grok_xai_async(full_prompt, system, grok_model, temp))
            if use_gemini:
                tasks.append(ask_gemini_async(full_prompt, system, gemini_model, temp))
            results: List[ProviderResult] = await asyncio.gather(*tasks) if tasks else []
            oks = [r for r in results if r.ok]
            return oks

        async def run_all():
            if debate_rounds and debate_rounds > 0:
                progress.progress(5, text="Débat: démarrage…")
                final_drafts, tr = await run_debate_async(
                    full_prompt, system, debate_rounds,
                    use_gpt, use_grok, use_gemini,
                    gpt_model, grok_model, gemini_model,
                    temp, progress=progress
                )
                out_entries = []
                for provider, draft in final_drafts.items():
                    out_entries.append({
                        "provider": provider,
                        "model": {"gpt": gpt_model, "grok": grok_model, "gemini": gemini_model}[provider],
                        "latency_s": 0.0,
                        "ok": True,
                        "error": None,
                        "output": _trim(draft)
                    })
                return out_entries, tr
            else:
                progress.progress(10, text="Appels LLM (parallèle)…")
                oks = await run_parallel_one_shot()
                progress.progress(70, text="Appels terminés.")
                if not oks:
                    return [], None
                out_entries = []
                for r in oks:
                    out_entries.append({
                        "provider": r.provider, "model": r.model, "latency_s": r.latency_s,
                        "ok": r.ok, "error": r.error, "output": _trim(r.output)
                    })
                return out_entries, None

        try:
            entries, transcript = asyncio.run(run_all())
        except Exception as e:
            progress.empty()
            st.error(f"Erreur d'exécution: {e}")
            st.stop()

        if not entries:
            progress.empty()
            st.error("Aucune réponse valide — vérifie tes clés API et tes modèles.")
            st.stop()

        progress.progress(85, text="Évaluation (juge)…")
        provider_for_judge = judge_provider if judge_kind == "llm" else "heuristic"
        try:
            scoreboard = judge_with_provider(prompt, entries, provider_for_judge, judge_model or None, weights)
        except Exception as e:
            st.warning(f"Juge LLM indisponible ({e}). Fallback heuristique.")
            scoreboard = judge_with_provider(prompt, entries, "heuristic", None, weights)
        progress.progress(100, text="Terminé.")
        time.sleep(0.2)
        progress.empty()

    # ========================= Display =========================
    st.success("Terminé ! ✅")

    tab_labels = [f"{e['provider'].upper()}" for e in entries] + ["📊 Scores"] + (["💬 Débat"] if transcript else [])
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

    if transcript:
        with tabs[-1]:
            st.write("Transcription du débat (extraits)")
            for rd in sorted(set(t["round"] for t in transcript)):
                st.markdown(f"#### Round {rd}")
                for t in [x for x in transcript if x["round"] == rd]:
                    with st.expander(f"{t['speaker'].upper()} ({t['model']})"):
                        st.text(t["text"])

    if save_btn:
        bundle = {
            "prompt": prompt,
            "system": system,
            "department": dep_key,
            "weights": weights,
            "entries": entries,
            "scoreboard": scoreboard,
            "rag_used": bool(rag_ctx),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        if transcript: bundle["transcript"] = transcript
        st.download_button("⬇️ Télécharger les résultats (JSON)",
                           data=json.dumps(bundle, ensure_ascii=False, indent=2),
                           file_name="results.json",
                           mime="application/json")
