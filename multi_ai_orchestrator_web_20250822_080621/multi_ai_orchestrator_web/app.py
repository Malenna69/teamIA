# -*- coding: utf-8 -*-
import os, json, time, re, datetime, math
import streamlit as st
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

# =============== Providers (OpenAI / xAI / Google) ===============
from openai import OpenAI
from google import genai
from google.genai import types as genai_types

TIMEOUT_S = 60

# ===== Optional PDF parsing =====
try:
    from pypdf import PdfReader  # pip install pypdf
    PYPDF_OK = True
except Exception:
    PYPDF_OK = False

# -------------------- Helpers --------------------
def _get_key(name: str) -> Optional[str]:
    if name in st.secrets: return st.secrets[name]
    return os.getenv(name)

def _get_temp(default_str: str, ui_value: Optional[float]) -> float:
    if ui_value is not None: return float(ui_value)
    try: return float(os.getenv("MODEL_TEMPERATURE", default_str))
    except Exception: return float(default_str)

def clamp(v, lo, hi): return max(lo, min(hi, v))

# -------------------- Providers --------------------
@dataclass
class ProviderResult:
    provider: str
    model: str
    output: str
    latency_s: float
    ok: bool
    error: Optional[str] = None

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
        txt = resp.choices[0].message.content.strip()
        return ProviderResult("gpt", model, txt, time.time()-t0, True)
    except Exception as e:
        return ProviderResult("gpt", model, "", time.time()-t0, False, str(e))

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
        txt = resp.choices[0].message.content.strip()
        return ProviderResult("grok", model, txt, time.time()-t0, True)
    except Exception as e:
        return ProviderResult("grok", model, "", time.time()-t0, False, str(e))

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
        res = client.models.generate_content(model=model, contents=prompt, config=cfg)
        txt = (res.text or "").strip()
        return ProviderResult("gemini", model, txt, time.time()-t0, True)
    except Exception as e:
        return ProviderResult("gemini", model, "", time.time()-t0, False, str(e))

# =============== Heuristic scoring & utilities ===============
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
    clamp10 = lambda x: max(0.0, min(10.0, x))
    return {"rigor": clamp10(rigor/5.0), "usefulness": clamp10(usefulness/5.0),
            "creativity": clamp10(creativity), "risk": clamp10(risk)}

def compute_total(s, weights):
    return (weights["rigor"] * s["rigor"] +
            weights["usefulness"] * s["usefulness"] +
            weights["creativity"] * s["creativity"] +
            weights["risk"] * (10 - s["risk"]))

def normalize_weights(r, u, c, k):
    total = max(1e-9, (r + u + c + k))
    return {"rigor": r/total, "usefulness": u/total, "creativity": c/total, "risk": k/total}

# =============== Judge (LLM / heuristic) ===============
def judge_with_provider(task: str, entries: list, provider: str, judge_model: Optional[str], weights: Dict[str,float]):
    if provider == "heuristic":
        scores = []
        for e in entries:
            s = heuristic_scores(e["output"]); s["provider"] = e["provider"]; scores.append(s)
        totals = [(s["provider"], compute_total(s, weights)) for s in scores]
        ranking = [p for p,_ in sorted(totals, key=lambda x: x[1], reverse=True)]
        return {"scores": scores, "weighted_ranking": ranking,
                "final_synthesis": "Synthèse non-LLM: utiliser le top-rank comme base et fusionner manuellement.",
                "action_plan": "- Prendre les points clés du meilleur score\n- Ajouter 2 actions du 2e meilleur\n- Lister les risques signalés"}

    bundle = {"task": task, "entries": entries, "weights": weights}
    prompt = (
        "You are an impartial evaluator.\n"
        "Return STRICT JSON only with keys: scores[], weighted_ranking[], final_synthesis, action_plan.\n"
        "Each item in scores[] must have keys: provider, rigor, usefulness, creativity, risk (0..10).\n"
        "Compute weighted_ranking using weights where risk DECREASES total by using (10-risk).\n\n"
        f"DATA:\n{json.dumps(bundle, ensure_ascii=False)}"
    )

    if provider in ("gpt","grok"):
        base_url = "https://api.x.ai/v1" if provider=="grok" else None
        api_key = _get_key("XAI_API_KEY") if provider=="grok" else _get_key("OPENAI_API_KEY")
        if not api_key: raise RuntimeError("Clé API manquante pour le juge sélectionné.")
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=judge_model or ("grok-4" if provider=="grok" else "gpt-4o-mini"),
            messages=[
                {"role":"system","content":"Return ONLY compact JSON per instructions. No preface."},
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
        res = client.models.generate_content(model=judge_model or "gemini-2.5-pro", contents=prompt, config=cfg)
        return json.loads(res.text)
    else:
        raise RuntimeError("Juge inconnu.")

# =============== Debate mode ===============
def role_system(base: Optional[str], style: str) -> str:
    base = base or "You are a respectful, concise expert."
    style = (style or "coopératif").lower()
    if style.startswith("coop"):
        tone = "Be collaborative: acknowledge strengths, propose merges, aim to synthesize the best of all drafts."
    elif style.startswith("cri"):
        tone = "Be sharply critical (but respectful): point out gaps, contradictions, and missing evidence with short actionable fixes."
    else:
        tone = "Be adversarial but professional: challenge assumptions, stress-test claims, and push for bolder, clearer, more verifiable output."
    extra = (
        " You are in a multi-agent debate. "
        "Each round, first write a brief CRITIQUE (<=120 words) of others' drafts, "
        "then produce an improved REVISION. "
        "Format:\nCRITIQUE: ...\nREVISION:\n..."
    )
    return f"{base} {extra} {tone}"

def prompt_for_round(task: str, self_name: str, drafts: Dict[str, str], round_idx: int, max_chars: int) -> str:
    others = {k:v for k,v in drafts.items() if k != self_name}
    def trim(txt: str, limit: int): 
        return txt if limit <= 0 else txt[:limit]
    blocks = [f"Task:\n{task}\n", f"Round: {round_idx}\n"]
    for name, draft in others.items():
        blocks.append(f"=== {name.upper()} CURRENT DRAFT ===\n{trim(draft, max_chars)}\n")
    blocks.append("Instructions:\n- Provide CRITIQUE (succinct, concrete) on gaps, errors, structure.\n- Then provide a clear, improved REVISION (self-contained).")
    return "\n".join(blocks)

def run_debate(task: str, system: Optional[str], rounds: int, style: str, max_chars: int,
               use_gpt: bool, use_grok: bool, use_gemini: bool,
               gpt_model: str, grok_model: str, gemini_model: str,
               temp: Optional[float]):
    participants = []
    if use_gpt: participants.append(("gpt", ask_openai_gpt, gpt_model))
    if use_grok: participants.append(("grok", ask_grok_xai, grok_model))
    if use_gemini: participants.append(("gemini", ask_gemini, gemini_model))

    drafts: Dict[str,str] = {}
    transcript: List[Dict] = []

    # Round 0
    for name, fn, model in participants:
        r = fn(task, role_system(system, style), model, temp)
        text = r.output if r.ok else f"[ERROR] {r.error}"
        drafts[name] = text
        transcript.append({"round": 0, "speaker": name, "model": model, "text": text})

    for rd in range(1, rounds+1):
        new_drafts: Dict[str,str] = {}
        for name, fn, model in participants:
            prompt = prompt_for_round(task, name, drafts, rd, max_chars)
            r = fn(prompt, role_system(system, style), model, temp)
            text = r.output if r.ok else f"[ERROR] {r.error}"
            rev = text
            up = text.upper()
            if "REVISION:" in up:
                idx = up.find("REVISION:")
                rev = text[idx+9:].strip()
            new_drafts[name] = rev or text
            transcript.append({"round": rd, "speaker": name, "model": model, "text": text})
        drafts = new_drafts
    return drafts, transcript

# =============== RAG (project files) ===============
def extract_text_from_file(upload) -> str:
    name = upload.name.lower()
    if name.endswith(".txt") or name.endswith(".md"):
        return upload.read().decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        if not PYPDF_OK: return "[PDF détecté mais pypdf non installé — ajoutez `pypdf` dans requirements.txt]"
        try:
            reader = PdfReader(upload)
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
        except Exception as e:
            return f"[Erreur PDF: {e}]"
    return "[Format non supporté — utilisez .txt .md .pdf]"

def chunk_text(txt: str, chunk_chars: int = 1200, overlap: int = 150) -> List[str]:
    if not txt: return []
    chunks = []
    i = 0
    n = len(txt)
    while i < n:
        j = min(n, i + chunk_chars)
        chunks.append(txt[i:j])
        i = j - overlap
        if i < 0: i = 0
    return chunks

def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    api_key = _get_key("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY manquant pour embeddings")
    client = OpenAI(api_key=api_key, timeout=TIMEOUT_S)
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def cosine_sim(a: List[float], b: List[float]) -> float:
    if not a or not b: return 0.0
    s = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na == 0 or nb == 0: return 0.0
    return s/(na*nb)

def build_rag_context(query: str, kb_store: Dict, top_k: int = 5) -> Tuple[str, List[Tuple[float,int]]]:
    if not kb_store or not kb_store.get("chunks"): return "", []
    # embed query
    qv = embed_texts([query])[0]
    sims = []
    for idx, vec in enumerate(kb_store["embeddings"]):
        sims.append((cosine_sim(qv, vec), idx))
    sims.sort(key=lambda x: x[0], reverse=True)
    picks = sims[:top_k]
    extracts = []
    for score, idx in picks:
        src = kb_store["meta"][idx]
        extracts.append(f"[{src}] {kb_store['chunks'][idx]}")
    ctx = "\n\n".join(extracts)
    return ctx, picks

# =============== Reports (Markdown/HTML) ===============
def build_markdown_report(prompt: str, system: str, entries: list, scoreboard: dict,
                          meta: dict, transcript: Optional[list], weights: Dict[str,float], kb_info: dict) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append(f"# Rapport Orchestrateur Multi-IA\n\n_Généré le {ts}_\n")
    lines.append("## 🎯 Prompt\n```\n" + (prompt or "") + "\n```\n")
    if system: lines.append("**System prompt :** " + system + "\n")
    lines.append("## ⚙️ Paramètres\n")
    lines.append(f"- Modèles: GPT={meta['gpt_model']}, Grok={meta['grok_model']}, Gemini={meta['gemini_model']}")
    lines.append(f"- Fournisseurs actifs: {', '.join(meta['active']) or '(aucun)'}")
    lines.append(f"- Température: {meta['temp']}  •  Débat: {meta['debate_rounds']} tours  •  Style: {meta['debate_style']}")
    lines.append(f"- Max chars par échange: {meta['max_chars']}")
    lines.append(f"- Poids: Rigor={weights['rigor']:.2f}  Usefulness={weights['usefulness']:.2f}  Creativity={weights['creativity']:.2f}  Risk={weights['risk']:.2f}\n")
    if kb_info:
        lines.append("## 📚 Connaissances (RAG)\n")
        lines.append(f"- Fichiers: {', '.join(kb_info.get('files', [])) or '(aucun)'}")
        lines.append(f"- Chunks: {kb_info.get('chunks_count', 0)}  •  Modèle d'embedding: text-embedding-3-small\n")
    lines.append("## 📊 Scores\n")
    rows = []
    for s in scoreboard.get("scores", []):
        total = compute_total(s, weights); rows.append((s["provider"], s["rigor"], s["usefulness"], s["creativity"], s["risk"], total))
    if rows:
        lines.append("| Provider | Rigor | Useful | Creat | Risk | Total |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in rows:
            lines.append(f"| {r[0]} | {r[1]:.2f} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f} | {r[5]:.2f} |")
    ranking = " > ".join(scoreboard.get("weighted_ranking", [])) or "(n/a)"
    lines.append("\n**Classement pondéré :** " + ranking + "\n")
    lines.append("## 🧩 Synthèse\n\n" + scoreboard.get("final_synthesis","(n/a)") + "\n")
    lines.append("## ✅ Plan d’action\n\n" + scoreboard.get("action_plan","(n/a)") + "\n")
    lines.append("\n---\n## 📦 Réponses par modèle\n")
    for e in entries:
        lines.append(f"\n### {e['provider'].upper()} • {e['model']} • {e.get('latency_s',0):.2f}s\n")
        lines.append("```\n" + (e["output"] or "") + "\n```\n")
    if transcript:
        lines.append("\n---\n## 💬 Débat (transcript)\n")
        for rd in sorted(set(t["round"] for t in transcript)):
            lines.append(f"\n### Round {rd}\n")
            for t in [x for x in transcript if x["round"] == rd]:
                lines.append(f"**{t['speaker'].upper()} ({t['model']})**\n\n```\n{t['text']}\n```\n")
    return "\n".join(lines)

def build_html_report(markdown_text: str, title: str="Rapport Orchestrateur"):
    safe = markdown_text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    return f"""<!doctype html>
<html lang="fr"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body{{font-family:ui-sans-serif,-apple-system,Segoe UI,Roboto,Arial;max-width:960px;margin:40px auto;line-height:1.5}}
h1,h2,h3{{line-height:1.2}}
pre{{background:#f7f7f9;padding:16px;border-radius:8px;overflow:auto}}
code{{font-family:ui-monospace,Menlo,Consolas,monospace}}
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ddd;padding:6px}} th{{background:#f0f0f0}}
</style>
</head><body>
<pre>{safe}</pre>
</body></html>"""

# =============== UI & STATE ===============
st.set_page_config(page_title="Team IA – ECOSWITCH (Web)", layout="wide")
st.title("🛠️ Team IA – ECOSWITCH")
st.caption("Mémoire projet • Playbooks • Docs & RAG • Débat multi-IA • Juge • Exports")

if "project_mem" not in st.session_state:
    st.session_state["project_mem"] = {
        "brief": "", "audience": "", "voice": "", "objectives": "", "constraints": ""
    }
if "kb_store" not in st.session_state:
    st.session_state["kb_store"] = {}  # {"chunks":[], "embeddings":[], "meta":[], "files":[]}

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("1) 🔐 Clés API")
    openai_key = st.text_input("OPENAI_API_KEY", value=_get_key("OPENAI_API_KEY") or "", type="password")
    xai_key = st.text_input("XAI_API_KEY", value=_get_key("XAI_API_KEY") or "", type="password")
    google_key = st.text_input("GOOGLE_API_KEY", value=_get_key("GOOGLE_API_KEY") or "", type="password")
    if openai_key: os.environ["OPENAI_API_KEY"] = openai_key
    if xai_key: os.environ["XAI_API_KEY"] = xai_key
    if google_key: os.environ["GOOGLE_API_KEY"] = google_key

    st.header("2) 🧠 Mémoire projet")
    m = st.session_state["project_mem"]
    m["brief"]       = st.text_area("Brief produit", m["brief"], height=80, key="mem_brief")
    m["audience"]    = st.text_area("Audience & ICP", m["audience"], height=60, key="mem_aud")
    m["voice"]       = st.text_area("Ton de marque", m["voice"], height=60, key="mem_voice")
    m["objectives"]  = st.text_area("Objectifs (OKR)", m["objectives"], height=60, key="mem_obj")
    m["constraints"] = st.text_area("Contraintes (tech/brand/legal)", m["constraints"], height=60, key="mem_cons")
    inject_mem = st.checkbox("Injecter la mémoire dans chaque tâche", value=True)

    colm1, colm2 = st.columns(2)
    with colm1:
        if st.button("💾 Télécharger mémoire"):
            st.download_button("Clique pour télécharger", data=json.dumps(m, ensure_ascii=False, indent=2),
                               file_name="ecoswitch_memory.json", mime="application/json", key="dl_mem_btn")
    with colm2:
        mem_upload = st.file_uploader("Charger mémoire (.json)", type=["json"], key="up_mem")
        if mem_upload:
            try:
                st.session_state["project_mem"] = json.load(mem_upload)
                st.success("Mémoire chargée.")
            except Exception as e:
                st.error(f"Erreur de chargement mémoire: {e}")

    st.header("3) 📚 Docs projet (RAG)")
    files = st.file_uploader("Uploader des fichiers (.txt .md .pdf)", type=["txt","md","pdf"], accept_multiple_files=True)
    if files:
        raw_texts, metas, file_names = [], [], []
        for up in files:
            txt = extract_text_from_file(up)
            chunks = chunk_text(txt, 1200, 150)
            meta_chunks = [f"{up.name}" for _ in chunks]
            raw_texts += chunks
            metas += meta_chunks
            file_names.append(up.name)
        try:
            vecs = embed_texts(raw_texts) if raw_texts else []
            st.session_state["kb_store"] = {"chunks": raw_texts, "embeddings": vecs, "meta": metas, "files": file_names}
            st.success(f"Indexé: {len(raw_texts)} passages depuis {len(file_names)} fichier(s).")
        except Exception as e:
            st.error(f"Embeddings échoués: {e}")
    use_rag = st.checkbox("Activer RAG (utiliser les docs pour enrichir les réponses)", value=False)
    rag_k = st.slider("Nb d'extraits (top-K)", 1, 10, 5)

    st.header("4) ⚙️ Paramètres IA")
    use_gpt   = st.checkbox("GPT (OpenAI)", value=True)
    use_grok  = st.checkbox("Grok (xAI)", value=True)
    use_gemini= st.checkbox("Gemini (Google)", value=True)
    gpt_model   = st.text_input("GPT model", "gpt-4o-mini")
    grok_model  = st.text_input("Grok model", "grok-4")
    gemini_model= st.text_input("Gemini model", "gemini-2.5-flash")
    temp = st.slider("Température", 0.0, 1.5, 0.6, 0.1)
    debate_rounds = st.number_input("Débat — nb de tours", 0, 5, 0, 1)
    debate_style  = st.selectbox("Style de débat", ["coopératif","critique","agressif"], index=0)
    max_chars     = st.number_input("Max chars/échange (0=illimité)", 0, 20000, 4000, 500)

    st.header("5) 🧮 Juge & Poids")
    judge_kind = st.selectbox("Type", ["llm","heuristic"], index=0)
    judge_provider = st.selectbox("Fournisseur juge (si llm)", ["gpt","grok","gemini"], index=0)
    judge_model = st.text_input("Modèle juge (optionnel)", "")
    w_rigor = st.slider("Poids rigueur", 0, 100, 40, 1)
    w_use  = st.slider("Poids utilité", 0, 100, 30, 1)
    w_crea = st.slider("Poids créativité", 0, 100, 20, 1)
    w_risk = st.slider("Poids risque (pénalise)", 0, 100, 10, 1)
    weights = normalize_weights(w_rigor, w_use, w_crea, w_risk)
    st.caption(f"Poids → R:{weights['rigor']:.2f} U:{weights['usefulness']:.2f} C:{weights['creativity']:.2f} K:{weights['risk']:.2f}")

# -------------------- PLAYBOOKS --------------------
st.subheader("📒 Playbooks ECOSWITCH (tâches types)")
PLAYBOOKS = {
    "Spec fonctionnelle (feature)": [
        "Rédige une spec fonctionnelle pour la feature « {feature} » : problème, objectifs, user stories, règles métier, contraintes, critères d’acceptation, risques, métriques de succès.",
        "Détaille l’API/événements/permissions nécessaires pour « {feature} » (schéma bref)."
    ],
    "Plan de démo salon (7–10 min)": [
        "Plan de démo pour salon : accroche, déroulé minute par minute, 3 visuels clés, interactions live, métriques temps réel, CTA stand & QR code.",
        "Prépare un script orateur (2 versions: 5 min & 10 min) et une checklist matériel."
    ],
    "Pitch investisseur (5 min)": [
        "Pitch 5 min : problème, solution, différenciation, traction (si dispo), business model, GTM, roadmap 12 mois, demande (ASK).",
        "Liste 6 Q/R d’objections investisseurs avec réponses concises et chiffrables."
    ],
    "GTM / Landing page": [
        "Rédige la section ‘hero’ d’une landing : headline, sub-head, 3 bénéfices avec preuve, CTA, social proof (placeholders).",
        "Écris 3 variantes d’emails cold (A/B/C) ciblant DAF / Ops / CTO pour ECOSWITCH."
    ],
    "Teardown concurrent": [
        "Fais un teardown de {competitor}: positionnement, pricing, segments, forces/faiblesses, angle de différenciation possible.",
        "Propose un contre-positionnement en 3 messages clés + 3 features ‘moat’."
    ],
    "Propositions de valeur (B2B)": [
        "Formule 5 propositions de valeur différenciantes (preuve opérable, métrique, objection, réponse).",
        "Convertis en 5 bullets slide-ready (une ligne chacun)."
    ]
}

col_pb1, col_pb2, col_pb3 = st.columns([2,2,1])
with col_pb1:
    pb_name = st.selectbox("Choisis un playbook", list(PLAYBOOKS.keys()), index=0)
with col_pb2:
    step_idx = st.number_input("Étape", 1, 2, 1, 1)
with col_pb3:
    add_pb = st.button("Charger l’étape")

# -------------------- PROMPTS --------------------
if "prompt" not in st.session_state: st.session_state["prompt"] = ""
if "system" not in st.session_state: st.session_state["system"] = "Tu es un comité d'experts (ingénierie, UX, marché). Style: clair, structuré, actionnable."

if add_pb:
    template = PLAYBOOKS[pb_name][step_idx-1]
    # Remplacements simples de variables connues
    template = template.replace("{feature}", "optimisation des pics de consommation (peak shaving)")
    template = template.replace("{competitor}", "Schneider EcoStruxure (exemple)")
    st.session_state["prompt"] = template

prompt = st.text_area("🧠 Prompt", height=160, key="prompt", placeholder="Décris la tâche ECOSWITCH…")
system = st.text_input("🗣️ System prompt (optionnel)", key="system")

# -------------------- ACTIONS --------------------
colA, colB, colC = st.columns([1,1,1])
with colA: run_btn = st.button("🚀 Lancer", type="primary")
with colB: clear_btn = st.button("🧽 Nettoyer")
with colC: save_local = st.checkbox("Activer exports (JSON/MD/HTML)", value=True)

if clear_btn:
    st.session_state["prompt"] = ""
    st.session_state["system"] = "Tu es un comité d'experts (ingénierie, UX, marché). Style: clair, structuré, actionnable."
    st.stop()

# -------------------- EXECUTION --------------------
entries = []; scoreboard = {}; transcript = None
meta = {
    "gpt_model": gpt_model, "grok_model": grok_model, "gemini_model": gemini_model,
    "active": [p for p,flag in (("gpt",use_gpt),("grok",use_grok),("gemini",use_gemini)) if flag],
    "temp": temp, "debate_rounds": debate_rounds, "debate_style": debate_style, "max_chars": max_chars
}

def assemble_task_prompt(user_prompt: str) -> str:
    blocks = []
    if inject_mem:
        pm = st.session_state["project_mem"]
        blocks.append("### CONTEXTE PROJET (Mémoire)\n"
                      f"- Brief: {pm.get('brief','')}\n"
                      f"- Audience: {pm.get('audience','')}\n"
                      f"- Ton: {pm.get('voice','')}\n"
                      f"- Objectifs: {pm.get('objectives','')}\n"
                      f"- Contraintes: {pm.get('constraints','')}\n")
    # RAG
    if use_rag and st.session_state["kb_store"].get("chunks"):
        ctx, picks = build_rag_context(user_prompt, st.session_state["kb_store"], top_k=rag_k)
        if ctx.strip():
            blocks.append("### EXTRAITS DOCS (RAG)\n" + ctx)
    blocks.append("### TÂCHE\n" + (user_prompt or ""))
    return "\n\n".join([b for b in blocks if b.strip()])

if run_btn:
    if not (prompt or "").strip():
        st.error("Merci d'écrire un prompt.")
    else:
        final_prompt = assemble_task_prompt(prompt)
        with st.spinner("Exécution en cours…"):
            if debate_rounds and debate_rounds > 0:
                final_drafts, transcript = run_debate(
                    final_prompt, system, debate_rounds, debate_style, max_chars,
                    use_gpt, use_grok, use_gemini,
                    gpt_model, grok_model, gemini_model, temp
                )
                for provider, draft in final_drafts.items():
                    entries.append({
                        "provider": provider,
                        "model": {"gpt": gpt_model, "grok": grok_model, "gemini": gemini_model}[provider],
                        "latency_s": 0.0, "ok": True, "error": None,
                        "output": draft if max_chars<=0 else draft[:15000]
                    })
            else:
                results = []
                if use_gpt:   results.append(ask_openai_gpt(final_prompt, system, gpt_model, temp))
                if use_grok:  results.append(ask_grok_xai(final_prompt, system, grok_model, temp))
                if use_gemini:results.append(ask_gemini(final_prompt, system, gemini_model, temp))
                ok = [r for r in results if r.ok]
                if not ok: st.error("Aucune réponse valide — vérifie tes clés API et tes modèles."); st.stop()
                for r in ok:
                    entries.append({
                        "provider": r.provider, "model": r.model, "latency_s": r.latency_s,
                        "ok": r.ok, "error": r.error,
                        "output": (r.output if max_chars<=0 else r.output[:15000])
                    })
            provider_for_judge = judge_provider if judge_kind == "llm" else "heuristic"
            scoreboard = judge_with_provider(prompt, entries, provider_for_judge, judge_model or None, weights)
            if not scoreboard.get("scores"):
                scores = []
                for e in entries:
                    s = heuristic_scores(e["output"]); s["provider"] = e["provider"]; scores.append(s)
                totals = [(s["provider"], compute_total(s, weights)) for s in scores]
                ranking = [p for p,_ in sorted(totals, key=lambda x: x[1], reverse=True)]
                scoreboard["scores"] = scores
                scoreboard.setdefault("weighted_ranking", ranking)
                scoreboard.setdefault("final_synthesis", "(Fallback) Synthèse heuristique.")
                scoreboard.setdefault("action_plan", "- Prendre les points clés du meilleur score\n- Ajouter 2 actions du 2e meilleur\n- Lister les risques signalés")
        st.success("Terminé !")

# -------------------- DISPLAY --------------------
if entries:
    tabs = st.tabs([f"{e['provider'].upper()}" for e in entries] + ["📊 Scores"] + (["💬 Débat"] if transcript else []) + (["📤 Export"] if save_local else []))

    for i, e in enumerate(entries):
        with tabs[i]:
            st.caption(f"Modèle: {e['model']} • Latence: {e.get('latency_s',0):.2f}s")
            st.text_area(f"Sortie – {e['provider'].upper()}", value=e["output"], height=350, key=f"out_{i}_{e['provider']}")

    with tabs[len(entries)]:
        rows = []
        for s in scoreboard.get("scores", []):
            rows.append({
                "provider": s["provider"],
                "rigor": round(s["rigor"],2),
                "usefulness": round(s["usefulness"],2),
                "creativity": round(s["creativity"],2),
                "risk": round(s["risk"],2),
                "total": round(compute_total(s, weights),2)
            })
        if rows:
            st.dataframe(rows, use_container_width=True)
            ranking = " > ".join(scoreboard.get("weighted_ranking", [])) or "(n/a)"
            st.markdown(f"**Classement pondéré :** {ranking}")
        st.subheader("Synthèse"); st.write(scoreboard.get("final_synthesis","(n/a)"))
        st.subheader("Plan d'action"); st.write(scoreboard.get("action_plan","(n/a)"))

    if transcript:
        with tabs[len(entries)+1]:
            st.write("Transcription du débat (extraits)")
            for rd in sorted(set(t["round"] for t in transcript)):
                st.markdown(f"#### Round {rd}")
                for t in [x for x in transcript if x["round"] == rd]:
                    with st.expander(f"Round {rd} — {t['speaker'].upper()} ({t['model']})"):
                        st.text(t["text"])

    if save_local:
        with tabs[-1]:
            kb = st.session_state.get("kb_store", {})
            kb_info = {"files": kb.get("files", []), "chunks_count": len(kb.get("chunks", []))}
            bundle = {"prompt": prompt, "system": system, "entries": entries, "scoreboard": scoreboard, "meta": meta, "weights": weights, "kb": kb_info}
            if transcript: bundle["transcript"] = transcript
            md = build_markdown_report(prompt, system, entries, scoreboard, meta, transcript, weights, kb_info)
            html = build_html_report(md, title="Rapport Team IA – ECOSWITCH")
            st.download_button("⬇️ JSON (résultats)", data=json.dumps(bundle, ensure_ascii=False, indent=2), file_name="results.json", mime="application/json", key="dl_json")
            st.download_button("⬇️ Markdown (.md)", data=md, file_name="rapport.md", mime="text/markdown", key="dl_md")
            st.download_button("⬇️ HTML (.html)", data=html, file_name="rapport.html", mime="text/html", key="dl_html")
