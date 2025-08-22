# -*- coding: utf-8 -*-
# =================== Team IA – ECOSWITCH (Web) — v2 optimisée ==================
# + Presets Départements (12)
# + Mémoire Projet & RAG (fallback sans embeddings si indisponible)
# + Débat multi-IA (GPT/Grok/Gemini) avec barre de progression
# + Juge (heuristique/LLM) et exports JSON/MD/HTML
# + Parallélisation des appels IA (ThreadPoolExecutor)
# + Keys Streamlit stables (pas de DuplicateElementId)
# ------------------------------------------------------------------------------
import os, json, time, re, datetime, math, random
import streamlit as st
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Team IA – ECOSWITCH", layout="wide")

# =============================== Providers ====================================
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

# ------------------------------ Helpers ---------------------------------------
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

def normalize_weights(r, u, c, k):
    total = max(1e-9, (r + u + c + k))
    return {"rigor": r/total, "usefulness": u/total, "creativity": c/total, "risk": k/total}

def clamp(v, lo, hi): return max(lo, min(hi, v))

# ---------------------------- Provider calls ----------------------------------
@dataclass
class ProviderResult:
    provider: str
    model: str
    output: str
    latency_s: float
    ok: bool
    error: Optional[str] = None

def _retry_call(fn, attempts=2, first_delay=0.8, backoff=1.7):
    last_e = None
    delay = first_delay
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_e = e
            time.sleep(delay + random.uniform(0, 0.3))
            delay *= backoff
    raise last_e

def ask_openai_gpt(prompt: str, system: Optional[str], model: str, temp: Optional[float]) -> ProviderResult:
    api_key = _get_key("OPENAI_API_KEY")
    if not api_key:
        return ProviderResult("gpt", model, "", 0.0, False, "OPENAI_API_KEY manquant")
    client = OpenAI(api_key=api_key, timeout=TIMEOUT_S)
    t0 = time.time()
    try:
        def _call():
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system","content": system or "You are a rigorous, neutral assistant."},
                    {"role":"user","content": prompt}
                ],
                temperature=_get_temp("0.7", temp)
            )
        resp = _retry_call(_call, attempts=2)
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
        def _call():
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system","content": system or "Be precise, insightful and practical."},
                    {"role":"user","content": prompt}
                ],
                temperature=_get_temp("0.8", temp)
            )
        resp = _retry_call(_call, attempts=2)
        txt = resp.choices[0].message.content.strip()
        return ProviderResult("grok", model, txt, time.time()-t0, True)
    except Exception as e:
        return ProviderResult("grok", model, "", time.time()-t0, False, str(e))

def ask_gemini(prompt: str, system: Optional[str], model: str, temp: Optional[float]) -> ProviderResult:
    api_key = _get_key("GOOGLE_API_KEY")
    if not api_key:
        return ProviderResult("gemini", model, "", 0.0, False, "GOOGLE_API_KEY manquant")
    t0 = time.time()
    try:
        client = genai.Client(api_key=api_key)
        cfg = genai_types.GenerateContentConfig(
            system_instruction=system or "You are a clear, structured, neutral expert.",
            temperature=_get_temp("0.6", temp),
        )
        def _call():
            return client.models.generate_content(model=model, contents=prompt, config=cfg)
        res = _retry_call(_call, attempts=2)
        txt = (res.text or "").strip()
        return ProviderResult("gemini", model, txt, time.time()-t0, True)
    except Exception as e:
        # Fallback (au cas où le SDK change)
        try:
            client = genai.Client(api_key=api_key)
            res = client.models.generate_content(model=model, contents=prompt)
            txt = (res.text or "").strip()
            return ProviderResult("gemini", model, txt, time.time()-t0, True)
        except Exception as e2:
            return ProviderResult("gemini", model, "", time.time()-t0, False, f"{e} / {e2}")

# ===================== Heuristic scoring & utilities ==========================
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

# ================================ Judge =======================================
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
        res = client.models.generate_content(
            model=judge_model or "gemini-2.5-pro",
            contents=prompt,
            config=cfg
        )
        return json.loads(res.text)
    else:
        raise RuntimeError("Juge inconnu.")

# ============================== Debate mode ===================================
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
    blocks.append(
        "Instructions:\n"
        "- Provide CRITIQUE (succinct, concrete) on gaps, errors, structure.\n"
        "- Then provide a clear, improved REVISION (self-contained)."
    )
    return "\n".join(blocks)

def run_debate(task: str, system: Optional[str], rounds: int, style: str, max_chars: int,
               use_gpt: bool, use_grok: bool, use_gemini: bool,
               gpt_model: str, grok_model: str, gemini_model: str,
               temp: Optional[float], progress_cb=None):
    participants = []
    if use_gpt: participants.append(("gpt", ask_openai_gpt, gpt_model))
    if use_grok: participants.append(("grok", ask_grok_xai, grok_model))
    if use_gemini: participants.append(("gemini", ask_gemini, gemini_model))

    drafts: Dict[str,str] = {}
    transcript: List[Dict] = []

    total_steps = (rounds + 1) * max(1, len(participants))
    done = 0

    # Round 0: initial drafts
    for name, fn, model in participants:
        r = fn(task, role_system(system, style), model, temp)
        text = r.output if r.ok else f"[ERROR] {r.error}"
        drafts[name] = text
        transcript.append({"round": 0, "speaker": name, "model": model, "text": text})
        done += 1
        if progress_cb: progress_cb(done / total_steps)

    # Debate rounds
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
            done += 1
            if progress_cb: progress_cb(done / total_steps)
        drafts = new_drafts
    return drafts, transcript

# ============================== RAG (docs) ====================================
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

def _keyword_fallback(query: str, chunks: List[str], meta: List[str], top_k: int) -> Tuple[str, List[Tuple[float,int]]]:
    terms = [w for w in re.findall(r"[a-zA-ZÀ-ÿ0-9]+", query.lower()) if len(w) > 2]
    scores = []
    for idx, ch in enumerate(chunks):
        txt = ch.lower()
        score = sum(txt.count(t) for t in terms)
        if "ecoswitch" in txt: score += 1  # petit boost contextuel
        scores.append((score, idx))
    scores.sort(key=lambda x: x[0], reverse=True)
    picks = [(float(s), i) for s,i in scores[:top_k]]
    extracts = []
    for _, idx in picks:
        src = meta[idx]
        extracts.append(f"[{src}] {chunks[idx]}")
    return "\n\n".join(extracts), picks

def build_rag_context(query: str, kb_store: Dict, top_k: int = 5) -> Tuple[str, List[Tuple[float,int]]]:
    if not kb_store or not kb_store.get("chunks"): return "", []
    try:
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
        return "\n\n".join(extracts), picks
    except Exception:
        # Fallback simple mots-clés si embeddings indisponibles
        return _keyword_fallback(query, kb_store["chunks"], kb_store["meta"], top_k)

# ============================ Reports (MD / HTML) =============================
def build_markdown_report(prompt: str, system: str, entries: list, scoreboard: dict,
                          meta: dict, transcript: Optional[list], weights: Dict[str,float], kb_info: dict) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append(f"# Rapport Orchestrateur Multi-IA\n\n_Généré le {ts}_\n")
    lines.append("## 🎯 Prompt\n```\n" + (prompt or "") + "\n```\n")
    if system:
        lines.append("**System prompt :** " + system + "\n")
    lines.append("## ⚙️ Paramètres\n")
    lines.append(f"- Modèles: GPT={meta['gpt_model']}, Grok={meta['grok_model']}, Gemini={meta['gemini_model']}")
    lines.append(f"- Fournisseurs actifs: {', '.join(meta['active']) or '(aucun)'}")
    lines.append(f"- Température: {meta['temp']}  •  Débat: {meta['debate_rounds']} tours  •  Style: {meta['debate_style']}")
    lines.append(f"- Max chars par échange: {meta['max_chars']}")
    lines.append(f"- Poids: Rigor={weights['rigor']:.2f}  Usefulness={weights['usefulness']:.2f}  Creativity={weights['creativity']:.2f}  Risk={weights['risk']:.2f}\n")
    if kb_info:
        lines.append("## 📚 Connaissances (RAG)\n")
        lines.append(f"- Fichiers: {', '.join(kb_info.get('files', [])) or '(aucun)'}")
        lines.append(f"- Chunks: {kb_info.get('chunks_count', 0)}  •  Modèle d'embedding: text-embedding-3-small (ou fallback mots-clés)\n")
    lines.append("## 📊 Scores\n")
    rows = []
    for s in scoreboard.get("scores", []):
        total = compute_total(s, weights)
        rows.append((s["provider"], s["rigor"], s["usefulness"], s["creativity"], s["risk"], total))
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

# ========================= Session defaults ==================================
for k, v in {
    "project_mem": {"brief":"", "audience":"", "voice":"", "objectives":"", "constraints":""},
    "kb_store": {},
    "prompt": "",
    "system": "Tu es un comité d'experts (ingénierie, UX, marché). Style: clair, structuré, actionnable.",
    # UI defaults
    "use_gpt": True, "use_grok": True, "use_gemini": True,
    "gpt_model": "gpt-4o-mini", "grok_model": "grok-4", "gemini_model": "gemini-2.5-flash",
    "temperature": 0.6, "debate_rounds": 0, "debate_style": "coopératif", "max_chars": 4000,
    "judge_kind": "llm", "judge_provider": "gpt", "judge_model": "",
    "w_rigor": 40, "w_use": 30, "w_crea": 20, "w_risk": 10,
    "inject_mem": True, "use_rag": False, "rag_k": 5,
    "prev_dept": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================ Departments presets =============================
DEPARTMENTS = {
    "general": {
        "label": "Général / CEO",
        "system": "Tu es le CEO et un board multidisciplinaire d’EcoSwitch. Donne une vision claire, synthèse exécutable et prochaines étapes.",
        "weights": {"rigor":0.40,"usefulness":0.30,"creativity":0.20,"risk":0.10},
        "providers":{"gpt":True,"grok":True,"gemini":True},
        "models":{"gpt":"gpt-4o-mini","grok":"grok-4","gemini":"gemini-2.5-flash"},
        "temp":0.7,"debate_rounds":0,"debate_style":"coopératif","max_chars":4000,
        "judge_kind":"llm","judge_provider":"gpt","judge_model":""},
    "marketing": {
        "label": "Marketing",
        "system": "Tu es le département marketing B2B d’EcoSwitch. Focus: acquisition, contenu persuasif, ROI data-driven. Style: dynamique, exemples concrets.",
        "weights": {"rigor":0.30,"usefulness":0.40,"creativity":0.20,"risk":0.10},
        "providers":{"gpt":True,"grok":True,"gemini":True},
        "models":{"gpt":"gpt-4o-mini","grok":"grok-4","gemini":"gemini-2.5-flash"},
        "temp":0.8,"debate_rounds":1,"debate_style":"coopératif","max_chars":5000,
        "judge_kind":"llm","judge_provider":"gpt","judge_model":""},
    "coding": {
        "label": "Coding / Dev",
        "system": "Tu es le département dev full-stack d’EcoSwitch. Python/JS, APIs, scalabilité cloud. Donne du code commenté, testable.",
        "weights": {"rigor":0.50,"usefulness":0.30,"creativity":0.10,"risk":0.10},
        "providers":{"gpt":True,"grok":True,"gemini":False},
        "models":{"gpt":"gpt-4o-mini","grok":"grok-4","gemini":"gemini-2.5-flash"},
        "temp":0.4,"debate_rounds":0,"debate_style":"critique","max_chars":8000,
        "judge_kind":"heuristic","judge_provider":"gpt","judge_model":""},
    "ingenierie": {
        "label": "Ingénierie / IoT",
        "system": "Tu es le département ingénierie d’EcoSwitch. IoT, optimisation systèmes, simulations. Donne diagrammes textuels et trade-offs.",
        "weights": {"rigor":0.45,"usefulness":0.35,"creativity":0.10,"risk":0.10},
        "providers":{"gpt":True,"grok":True,"gemini":True},
        "models":{"gpt":"gpt-4o-mini","grok":"grok-4","gemini":"gemini-2.5-flash"},
        "temp":0.6,"debate_rounds":2,"debate_style":"critique","max_chars":6000,
        "judge_kind":"llm","judge_provider":"gpt","judge_model":""},
    "thermie-batiment": {
        "label": "Thermie du Bâtiment",
        "system": "Tu es le département thermique d’EcoSwitch. HVAC, RE2020, isolation. Donne formules et métriques énergétiques.",
        "weights": {"rigor":0.50,"usefulness":0.25,"creativity":0.15,"risk":0.10},
        "providers":{"gpt":False,"grok":True,"gemini":True},
        "models":{"gpt":"gpt-4o-mini","grok":"grok-4","gemini":"gemini-2.5-flash"},
        "temp":0.5,"debate_rounds":1,"debate_style":"coopératif","max_chars":7000,
        "judge_kind":"heuristic","judge_provider":"gpt","judge_model":""},
    "product": {
        "label": "Product Management",
        "system": "Tu es le département product. Lifecycle, user stories, MVP. Donne roadmap structurée et priorisation claire.",
        "weights": {"rigor":0.35,"usefulness":0.40,"creativity":0.15,"risk":0.10},
        "providers":{"gpt":True,"grok":True,"gemini":True},
        "models":{"gpt":"gpt-4o-mini","grok":"grok-4","gemini":"gemini-2.5-flash"},
        "temp":0.7,"debate_rounds":1,"debate_style":"coopératif","max_chars":5000,
        "judge_kind":"llm","judge_provider":"gpt","judge_model":""},
    "ux-ui": {
        "label": "UX / UI",
        "system": "Tu es le département UX/UI. Wireframes textuels, user journeys, accessibilité. Style: concret, gamification utile.",
        "weights": {"rigor":0.30,"usefulness":0.30,"creativity":0.30,"risk":0.10},
        "providers":{"gpt":True,"grok":True,"gemini":True},
        "models":{"gpt":"gpt-4o-mini","grok":"grok-4","gemini":"gemini-2.5-flash"},
        "temp":0.8,"debate_rounds":1,"debate_style":"coopératif","max_chars":4000,
        "judge_kind":"heuristic","judge_provider":"gpt","judge_model":""},
    "finance": {
        "label": "Finance & Pricing",
        "system": "Tu es le département finance. Pricing SaaS, ROI models, forecasts. Donne tableaux simples et hypothèses claires.",
        "weights": {"rigor":0.50,"usefulness":0.30,"creativity":0.10,"risk":0.10},
        "providers":{"gpt":True,"grok":True,"gemini":False},
        "models":{"gpt":"gpt-4o-mini","grok":"grok-4","gemini":"gemini-2.5-flash"},
        "temp":0.5,"debate_rounds":0,"debate_style":"critique","max_chars":6000,
        "judge_kind":"heuristic","judge_provider":"gpt","judge_model":""},
    "legal": {
        "label": "Legal & Compliance",
        "system": "Tu es le département legal d’EcoSwitch. GDPR, contrats B2B, ESG/CSRD. Donne checklists, risques et limites. Ne cite pas de loi si incertain.",
        "weights": {"rigor":0.60,"usefulness":0.25,"creativity":0.05,"risk":0.10},
        "providers":{"gpt":True,"grok":True,"gemini":True},
        "models":{"gpt":"gpt-4o-mini","grok":"grok-4","gemini":"gemini-2.5-flash"},
        "temp":0.3,"debate_rounds":1,"debate_style":"critique","max_chars":5000,
        "judge_kind":"heuristic","judge_provider":"gpt","judge_model":""},
    "operations": {
        "label": "Operations / Scaling",
        "system": "Tu es le département ops. Cloud scaling, sécurité, monitoring. Donne plans de déploiement, SLO/SLI.",
        "weights": {"rigor":0.45,"usefulness":0.35,"creativity":0.10,"risk":0.10},
        "providers":{"gpt":True,"grok":True,"gemini":True},
        "models":{"gpt":"gpt-4o-mini","grok":"grok-4","gemini":"gemini-2.5-flash"},
        "temp":0.6,"debate_rounds":1,"debate_style":"critique","max_chars":6000,
        "judge_kind":"llm","judge_provider":"gpt","judge_model":""},
    "sustainability": {
        "label": "Sustainability R&D",
        "system": "Tu es le département R&D sustainability. Tendances ENR, ML prédictif pour l’énergie/carbone. Donne idées testables.",
        "weights": {"rigor":0.40,"usefulness":0.25,"creativity":0.25,"risk":0.10},
        "providers":{"gpt":True,"grok":True,"gemini":True},
        "models":{"gpt":"gpt-4o-mini","grok":"grok-4","gemini":"gemini-2.5-flash"},
        "temp":0.7,"debate_rounds":2,"debate_style":"coopératif","max_chars":7000,
        "judge_kind":"llm","judge_provider":"gpt","judge_model":""},
    "sales": {
        "label": "Sales & Customer Success",
        "system": "Tu es le département sales. Scripts de vente, onboarding, feedback loops. Donne role-plays et métriques.",
        "weights": {"rigor":0.30,"usefulness":0.40,"creativity":0.20,"risk":0.10},
        "providers":{"gpt":True,"grok":True,"gemini":True},
        "models":{"gpt":"gpt-4o-mini","grok":"grok-4","gemini":"gemini-2.5-flash"},
        "temp":0.7,"debate_rounds":1,"debate_style":"coopératif","max_chars":5000,
        "judge_kind":"heuristic","judge_provider":"gpt","judge_model":""},
}

def _apply_department(dept_key: str):
    d = DEPARTMENTS[dept_key]
    st.session_state["system"] = d["system"]
    st.session_state["w_rigor"] = int(d["weights"]["rigor"]*100)
    st.session_state["w_use"]   = int(d["weights"]["usefulness"]*100)
    st.session_state["w_crea"]  = int(d["weights"]["creativity"]*100)
    st.session_state["w_risk"]  = int(d["weights"]["risk"]*100)
    st.session_state["use_gpt"]    = bool(d["providers"]["gpt"])
    st.session_state["use_grok"]   = bool(d["providers"]["grok"])
    st.session_state["use_gemini"] = bool(d["providers"]["gemini"])
    st.session_state["gpt_model"]   = d["models"]["gpt"]
    st.session_state["grok_model"]  = d["models"]["grok"]
    st.session_state["gemini_model"]= d["models"]["gemini"]
    st.session_state["temperature"]   = float(d["temp"])
    st.session_state["debate_rounds"] = int(d["debate_rounds"])
    st.session_state["debate_style"]  = d["debate_style"]
    st.session_state["max_chars"]     = int(d["max_chars"])
    st.session_state["judge_kind"]     = d["judge_kind"]
    st.session_state["judge_provider"] = d["judge_provider"]
    st.session_state["judge_model"]    = d["judge_model"]

# ================================== UI ========================================
st.title("🛠️ Team IA – ECOSWITCH")
st.caption("Mémoire projet • Docs & RAG • Débat multi-IA (GPT/Grok/Gemini) • Juge • Exports")

with st.sidebar:
    st.header("1) 🔐 Clés API")
    openai_key = st.text_input("OPENAI_API_KEY", value=_get_key("OPENAI_API_KEY") or "", type="password", key="OPENAI_API_KEY_UI")
    xai_key    = st.text_input("XAI_API_KEY",    value=_get_key("XAI_API_KEY") or "",    type="password", key="XAI_API_KEY_UI")
    google_key = st.text_input("GOOGLE_API_KEY", value=_get_key("GOOGLE_API_KEY") or "", type="password", key="GOOGLE_API_KEY_UI")
    if openai_key: os.environ["OPENAI_API_KEY"] = openai_key
    if xai_key:    os.environ["XAI_API_KEY"]    = xai_key
    if google_key: os.environ["GOOGLE_API_KEY"] = google_key

    st.header("2) 🧠 Mémoire projet")
    m = st.session_state["project_mem"]
    m["brief"]       = st.text_area("Brief produit", m["brief"], height=80, key="mem_brief")
    m["audience"]    = st.text_area("Audience & ICP", m["audience"], height=60, key="mem_aud")
    m["voice"]       = st.text_area("Ton de marque", m["voice"], height=60, key="mem_voice")
    m["objectives"]  = st.text_area("Objectifs (OKR)", m["objectives"], height=60, key="mem_obj")
    m["constraints"] = st.text_area("Contraintes (tech/brand/legal)", m["constraints"], height=60, key="mem_cons")
    st.session_state["inject_mem"] = st.checkbox("Injecter la mémoire dans chaque tâche", value=st.session_state["inject_mem"], key="inject_mem_cb")
    st.download_button("⬇️ Exporter la mémoire (.json)", data=json.dumps(m, ensure_ascii=False, indent=2),
                       file_name="ecoswitch_memory.json", mime="application/json", key="dl_mem")
    mem_upload = st.file_uploader("Charger mémoire (.json)", type=["json"], key="up_mem")
    if mem_upload:
        try:
            st.session_state["project_mem"] = json.load(mem_upload)
            st.success("Mémoire chargée.")
        except Exception as e:
            st.error(f"Erreur de chargement mémoire: {e}")

    st.header("3) 📚 Docs projet (RAG)")
    files = st.file_uploader("Uploader des fichiers (.txt .md .pdf)", type=["txt","md","pdf"], accept_multiple_files=True, key="files_up")
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
            st.warning(f"Embeddings indisponibles → fallback mots-clés. Détail: {e}")
            st.session_state["kb_store"] = {"chunks": raw_texts, "embeddings": [], "meta": metas, "files": file_names}

    st.session_state["use_rag"] = st.checkbox("Activer RAG (utiliser les docs)", value=st.session_state["use_rag"], key="use_rag_cb")
    st.session_state["rag_k"]  = st.slider("Nb d'extraits (top-K)", 1, 10, st.session_state["rag_k"], key="rag_k_sl")

    st.header("4) 🧩 Département (preset)")
    dept_keys = list(DEPARTMENTS.keys())
    dept_labels = [DEPARTMENTS[k]["label"] for k in dept_keys]
    sel_idx = st.selectbox("Choisis un département", list(range(len(dept_labels))), format_func=lambda i: dept_labels[i], index=0, key="dept_select")
    # Auto-apply preset si changement
    if st.session_state["prev_dept"] != sel_idx:
        chosen_key = dept_keys[sel_idx]
        _apply_department(chosen_key)
        st.session_state["prev_dept"] = sel_idx
        st.toast(f"Preset « {DEPARTMENTS[chosen_key]['label']} » appliqué.", icon="✅")

    st.header("5) ⚙️ Paramètres IA")
    st.session_state["use_gpt"]    = st.checkbox("GPT (OpenAI)", value=st.session_state["use_gpt"], key="use_gpt_cb")
    st.session_state["use_grok"]   = st.checkbox("Grok (xAI)",   value=st.session_state["use_grok"], key="use_grok_cb")
    st.session_state["use_gemini"] = st.checkbox("Gemini (Google)", value=st.session_state["use_gemini"], key="use_gemini_cb")

    st.session_state["gpt_model"]    = st.text_input("GPT model",    value=st.session_state["gpt_model"], key="gpt_model_in")
    st.session_state["grok_model"]   = st.text_input("Grok model",   value=st.session_state["grok_model"], key="grok_model_in")
    st.session_state["gemini_model"] = st.text_input("Gemini model", value=st.session_state["gemini_model"], key="gemini_model_in")

    st.session_state["temperature"]   = st.slider("Température", 0.0, 1.5, st.session_state["temperature"], 0.1, key="temp_sl")
    st.session_state["debate_rounds"] = st.number_input("Débat — nb de tours", 0, 5, st.session_state["debate_rounds"], 1, key="debate_nb")
    st.session_state["debate_style"]  = st.selectbox("Style de débat", ["coopératif","critique","agressif"],
                                                     index=["coopératif","critique","agressif"].index(st.session_state["debate_style"]), key="debate_style_sel")
    st.session_state["max_chars"]     = st.number_input("Max chars/échange (0=illimité)", 0, 20000, st.session_state["max_chars"], 500, key="max_chars_nb")

    st.header("6) 🧮 Juge & Poids")
    st.session_state["judge_kind"]     = st.selectbox("Type", ["llm","heuristic"], index=["llm","heuristic"].index(st.session_state["judge_kind"]), key="judge_kind_sel")
    st.session_state["judge_provider"] = st.selectbox("Fournisseur juge (si llm)", ["gpt","grok","gemini"],
                                                      index=["gpt","grok","gemini"].index(st.session_state["judge_provider"]), key="judge_provider_sel")
    st.session_state["judge_model"]    = st.text_input("Modèle juge (optionnel)", value=st.session_state["judge_model"], key="judge_model_in")

    st.session_state["w_rigor"] = st.slider("Rigueur", 0, 100, st.session_state["w_rigor"], 1, key="w_rigor_sl")
    st.session_state["w_use"]   = st.slider("Utilité", 0, 100, st.session_state["w_use"], 1, key="w_use_sl")
    st.session_state["w_crea"]  = st.slider("Créativité", 0, 100, st.session_state["w_crea"], 1, key="w_crea_sl")
    st.session_state["w_risk"]  = st.slider("Risque (pénalise)", 0, 100, st.session_state["w_risk"], 1, key="w_risk_sl")

    _w = normalize_weights(st.session_state["w_rigor"], st.session_state["w_use"], st.session_state["w_crea"], st.session_state["w_risk"])
    st.caption(f"Poids → R:{_w['rigor']:.2f} U:{_w['usefulness']:.2f} C:{_w['creativity']:.2f} K:{_w['risk']:.2f}")

# ----------------------------- Playbooks simples ------------------------------
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
    "Propositions de valeur (B2B)": [
        "Formule 5 propositions de valeur différenciantes (preuve opérable, métrique, objection, réponse).",
        "Convertis en 5 bullets slide-ready (une ligne chacun)."
    ],
}
col_pb1, col_pb2, col_pb3 = st.columns([2,2,1])
with col_pb1:
    pb_name = st.selectbox("Choisis un playbook", list(PLAYBOOKS.keys()), index=0, key="pb_sel")
with col_pb2:
    feature_name = st.text_input("Variable {feature}", "optimisation des pics de consommation (peak shaving)", key="pb_feature")
with col_pb3:
    step_idx = st.number_input("Étape", 1, 2, 1, 1, key="pb_step")

if st.button("Charger l’étape", key="pb_load"):
    template = PLAYBOOKS[pb_name][int(step_idx)-1]
    st.session_state["prompt"] = template.replace("{feature}", feature_name)

# ------------------------------- Prompt zone ----------------------------------
prompt = st.text_area("🧠 Prompt", height=160, key="prompt", placeholder="Décris la tâche ECOSWITCH…")
system = st.text_input("🗣️ System prompt (optionnel)", key="system")

# ------------------------------- Actions --------------------------------------
colA, colB, colC = st.columns([1,1,1])
with colA: run_btn = st.button("🚀 Lancer", type="primary", key="run_btn")
with colB: clear_btn = st.button("🧽 Nettoyer", key="clear_btn")
with colC: save_local = st.checkbox("Activer exports (JSON/MD/HTML)", value=True, key="save_local_cb")

if clear_btn:
    st.session_state["prompt"] = ""
    st.session_state["system"] = "Tu es un comité d'experts (ingénierie, UX, marché). Style: clair, structuré, actionnable."
    st.stop()

# -------------------------- Assemble prompt (mémoire/RAG) ---------------------
def assemble_task_prompt(user_prompt: str) -> str:
    blocks = []
    if st.session_state["inject_mem"]:
        pm = st.session_state["project_mem"]
        blocks.append("### CONTEXTE PROJET (Mémoire)\n"
                      f"- Brief: {pm.get('brief','')}\n"
                      f"- Audience: {pm.get('audience','')}\n"
                      f"- Ton: {pm.get('voice','')}\n"
                      f"- Objectifs: {pm.get('objectives','')}\n"
                      f"- Contraintes: {pm.get('constraints','')}\n")
    if st.session_state["use_rag"] and st.session_state["kb_store"].get("chunks"):
        ctx, picks = build_rag_context(user_prompt, st.session_state["kb_store"], top_k=st.session_state["rag_k"])
        if ctx.strip():
            blocks.append("### EXTRAITS DOCS (RAG)\n" + ctx)
    blocks.append("### TÂCHE\n" + (user_prompt or ""))
    return "\n\n".join([b for b in blocks if b.strip()])

# ------------------------------- Execution ------------------------------------
entries = []; scoreboard = {}; transcript = None
meta = {
    "gpt_model": st.session_state["gpt_model"],
    "grok_model": st.session_state["grok_model"],
    "gemini_model": st.session_state["gemini_model"],
    "active": [p for p,flag in (("gpt",st.session_state["use_gpt"]),("grok",st.session_state["use_grok"]),("gemini",st.session_state["use_gemini"])) if flag],
    "temp": st.session_state["temperature"],
    "debate_rounds": st.session_state["debate_rounds"],
    "debate_style": st.session_state["debate_style"],
    "max_chars": st.session_state["max_chars"]
}
weights = normalize_weights(st.session_state["w_rigor"], st.session_state["w_use"], st.session_state["w_crea"], st.session_state["w_risk"])

def _run_selected_providers(prompt_final: str, system: str):
    calls = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = []
        if st.session_state["use_gpt"]:
            futs.append(ex.submit(ask_openai_gpt, prompt_final, system, st.session_state["gpt_model"], st.session_state["temperature"]))
        if st.session_state["use_grok"]:
            futs.append(ex.submit(ask_grok_xai, prompt_final, system, st.session_state["grok_model"], st.session_state["temperature"]))
        if st.session_state["use_gemini"]:
            futs.append(ex.submit(ask_gemini, prompt_final, system, st.session_state["gemini_model"], st.session_state["temperature"]))
        for f in as_completed(futs):
            calls.append(f.result())
    # Stabiliser l’ordre des onglets
    ordered = []
    for name in ["gpt","grok","gemini"]:
        for r in calls:
            if r.provider == name and r.ok:
                ordered.append(r)
    return ordered

if run_btn:
    if not (prompt or "").strip():
        st.error("Merci d'écrire un prompt.")
    else:
        final_prompt = assemble_task_prompt(prompt)
        with st.spinner("Exécution en cours…"):
            if st.session_state["debate_rounds"] and st.session_state["debate_rounds"] > 0:
                prog = st.progress(0.0)
                def _cb(p): prog.progress(min(1.0, max(0.0, p)))
                final_drafts, transcript = run_debate(
                    final_prompt, system, st.session_state["debate_rounds"], st.session_state["debate_style"], st.session_state["max_chars"],
                    st.session_state["use_gpt"], st.session_state["use_grok"], st.session_state["use_gemini"],
                    st.session_state["gpt_model"], st.session_state["grok_model"], st.session_state["gemini_model"],
                    st.session_state["temperature"], progress_cb=_cb
                )
                for provider, draft in final_drafts.items():
                    entries.append({
                        "provider": provider,
                        "model": {"gpt": st.session_state["gpt_model"], "grok": st.session_state["grok_model"], "gemini": st.session_state["gemini_model"]}[provider],
                        "latency_s": 0.0, "ok": True, "error": None,
                        "output": draft if st.session_state["max_chars"]<=0 else draft[:15000]
                    })
            else:
                results = _run_selected_providers(final_prompt, system)
                if not results:
                    st.error("Aucune réponse valide — vérifie tes clés API et tes modèles.")
                    st.stop()
                for r in results:
                    entries.append({
                        "provider": r.provider, "model": r.model, "latency_s": r.latency_s,
                        "ok": r.ok, "error": r.error,
                        "output": (r.output if st.session_state["max_chars"]<=0 else r.output[:15000])
                    })
            provider_for_judge = st.session_state["judge_provider"] if st.session_state["judge_kind"] == "llm" else "heuristic"
            scoreboard = judge_with_provider(prompt, entries, provider_for_judge, st.session_state["judge_model"] or None, weights)
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

# -------------------------------- Display -------------------------------------
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
            st.download_button("⬇️ JSON (résultats)", data=json.dumps(bundle, ensure_ascii=False, indent=2),
                               file_name="results.json", mime="application/json", key="dl_json")
            st.download_button("⬇️ Markdown (.md)", data=md, file_name="rapport.md",
                               mime="text/markdown", key="dl_md")
            st.download_button("⬇️ HTML (.html)", data=html, file_name="rapport.html",
                               mime="text/html", key="dl_html")
