# -*- coding: utf-8 -*-
# =================== Team IA – ECOSWITCH (Web) — v3.1 avec Décideur Auto ===================
# Orchestrateur multi-IA (GPT/Grok/Gemini) + Débat + Juge + RAG + Exports
# + Outils complémentaires (Simulation, Code & Tests, Analyse marché, PM)
# + Décideur Auto 10/10 (multi-agents, plan-execute, XAI, fallback heuristique)
# -----------------------------------------------------------------------------------
import os, json, time, re, datetime, math, random, subprocess, sys, textwrap
import streamlit as st
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Team IA – ECOSWITCH", layout="wide")

# ============================ Dépendances tierces =============================
from openai import OpenAI
from google import genai
from google.genai import types as genai_types
from tenacity import retry, stop_after_attempt, wait_exponential

# RAG / PDF
try:
    from pypdf import PdfReader
    PYPDF_OK = True
except Exception:
    PYPDF_OK = False

# Simulation/plots/tabulaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Web fetch (Analyse marché)
try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_OK = True
except Exception:
    REQUESTS_OK = False

TIMEOUT_S = 60

# =============================== Helpers communs ===============================
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

# =============================== Providers LLM ================================
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
    for _ in range(attempts):
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
        try:
            client = genai.Client(api_key=api_key)
            res = client.models.generate_content(model=model, contents=prompt)
            txt = (res.text or "").strip()
            return ProviderResult("gemini", model, txt, time.time()-t0, True)
        except Exception as e2:
            return ProviderResult("gemini", model, "", time.time()-t0, False, f"{e} / {e2}")

# =========================== Heuristics & Judge ===============================
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

# ============================== Débat multi-IA =================================
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
    def trim(txt: str, limit: int): return txt if limit <= 0 else txt[:limit]
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

    for name, fn, model in participants:
        r = fn(task, role_system(system, style), model, temp)
        text = r.output if r.ok else f"[ERROR] {r.error}"
        drafts[name] = text
        transcript.append({"round": 0, "speaker": name, "model": model, "text": text})
        done += 1
        if progress_cb: progress_cb(done / total_steps)

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

# ============================== RAG (docs locaux) =============================
def extract_text_from_file(upload) -> str:
    name = upload.name.lower()
    if name.endswith(".txt") or name.endswith(".md"):
        return upload.read().decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        if not PYPDF_OK: return "[PDF détecté mais pypdf non installé — ajoutez `pypdf`]"
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
        if "ecoswitch" in txt: score += 1
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
        return _keyword_fallback(query, kb_store["chunks"], kb_store["meta"], top_k)

def get_rag_ctx_for_prompt(user_prompt: str) -> str:
    kb = st.session_state.get("kb_store", {})
    if st.session_state.get("use_rag") and kb.get("chunks"):
        ctx, _ = build_rag_context(user_prompt, kb, top_k=st.session_state.get("rag_k",5))
        return ctx
    return ""

# ============================== Reports (MD/HTML) =============================
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
        lines.append(f"- Chunks: {kb_info.get('chunks_count', 0)}  •  Embeddings: text-embedding-3-small (ou fallback mots-clés)\n")
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

# ========================= State (mémoire & defaults) =========================
for k, v in {
    "project_mem": {"brief":"", "audience":"", "voice":"", "objectives":"", "constraints":"", "decisions_history":[]},
    "kb_store": {},
    "prompt": "",
    "system": "Tu es un comité d'experts (ingénierie, UX, marché). Style: clair, structuré, actionnable.",
    "use_gpt": True, "use_grok": True, "use_gemini": True,
    "gpt_model": "gpt-4o-mini", "grok_model": "grok-4", "gemini_model": "gemini-2.5-flash",
    "temperature": 0.6, "debate_rounds": 0, "debate_style": "coopératif", "max_chars": 4000,
    "judge_kind": "llm", "judge_provider": "gpt", "judge_model": "",
    "w_rigor": 40, "w_use": 30, "w_crea": 20, "w_risk": 10,
    "inject_mem": True, "use_rag": False, "rag_k": 5,
    "last_decision": None, "criteria_text": "Prioriser faisabilité, coût bas, impact éco haut",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================ UI — En-tête commun =============================
st.title("🛠️ Team IA – ECOSWITCH")
st.caption("Orchestrateur multi-IA • Mémoire & RAG • Débat • Juge • Exports • Outils • Décideur Auto")

with st.sidebar:
    st.header("🔐 Clés API")
    openai_key = st.text_input("OPENAI_API_KEY", value=_get_key("OPENAI_API_KEY") or "", type="password", key="OPENAI_API_KEY_UI")
    xai_key    = st.text_input("XAI_API_KEY",    value=_get_key("XAI_API_KEY") or "",    type="password", key="XAI_API_KEY_UI")
    google_key = st.text_input("GOOGLE_API_KEY", value=_get_key("GOOGLE_API_KEY") or "", type="password", key="GOOGLE_API_KEY_UI")
    if openai_key: os.environ["OPENAI_API_KEY"] = openai_key
    if xai_key:    os.environ["XAI_API_KEY"]    = xai_key
    if google_key: os.environ["GOOGLE_API_KEY"] = google_key

# ======================== Onglets ========================
tab_orch, tab_sim, tab_code, tab_market, tab_pm = st.tabs(
    ["🧠 Orchestrateur", "🔬 Simulation énergétique", "🧪 Code & Tests", "📈 Analyse marché", "📋 PM / Kanban"]
)

# ===================== DÉCIDEUR AUTO (module) =====================
def _provider_order() -> List[str]:
    order = []
    if st.session_state.get("use_grok"): order.append("grok")
    if st.session_state.get("use_gpt"): order.append("gpt")
    if st.session_state.get("use_gemini"): order.append("gemini")
    if not order: order = ["gpt"]
    return order

def _llm_json_call(prompt: str, system: str, temp: float = 0.4) -> Dict[str, Any]:
    """Essaie Grok -> GPT -> Gemini jusqu'à obtenir du JSON parsable."""
    errs = []
    for p in _provider_order():
        if p == "grok":
            r = ask_grok_xai(prompt, system, st.session_state["grok_model"], temp)
        elif p == "gpt":
            r = ask_openai_gpt(prompt, system, st.session_state["gpt_model"], temp)
        else:
            r = ask_gemini(prompt, system, st.session_state["gemini_model"], temp)
        if r.ok and r.output:
            txt = r.output.strip()
            try:
                # isoler éventuel JSON dans du texte
                s = txt.find("{"); e = txt.rfind("}")
                if s != -1 and e != -1 and e > s:
                    return json.loads(txt[s:e+1])
                return json.loads(txt)
            except Exception as je:
                errs.append(f"{p}: parse JSON fail ({je})")
                continue
        else:
            errs.append(f"{p}: {r.error}")
    raise RuntimeError(" / ".join(errs) or "Aucun provider n'a répondu")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=3, max=10))
def decideur_auto(entries: List[Dict], weights: Dict[str, float], user_criteria: str,
                  project_mem: Dict = None, rag_ctx: str = "", max_agents: int = 3, max_loops: int = 2) -> Dict[str, Any]:
    """
    Agent autonome multi-étapes:
      1) Décomposition des tâches (scoring multi-critères + choix)
      2) Multi-agents (Technique / Business / ESG)
      3) Boucle plan-execute si ambiguïtés
      4) Sortie JSON: scores, choix, explication (XAI), plan[], alternatives[]
      5) Fallback heuristique si tout échoue
    """
    if not entries:
        return {"erreur": "Aucun output à analyser."}

    # Contexte
    context = {
        "project_mem": project_mem or {},
        "rag_extracts": rag_ctx or "",
        "weights": weights,
        "criteria": user_criteria
    }

    # Définition des agents
    focuses = ["Technique/faisabilité", "Business/ROI", "Éco/ESG"]
    focuses = focuses[:max_agents]

    agent_results: Dict[str, Any] = {}
    for i, focus in enumerate(focuses, start=1):
        prompt_agent = (
            "Tu es un agent décisionnel expert, objectif et complet.\n"
            "Retourne UNIQUEMENT un JSON compact avec ces clés:\n"
            "scores (liste d'objets par option, avec rigor,usefulness,creativity,risk,faisabilite,impact_eco,roi en 0..10),\n"
            "choix (string: identifiant de l'option gagnante — provider name),\n"
            "explication (string courte expliquant le pourquoi),\n"
            "plan (liste de 5-10 étapes actionnables),\n"
            "alternatives (liste de 2-3 options de secours),\n"
            "raffinement_needed (bool).\n\n"
            f"FOCUS AGENT: {focus}\n"
            f"CONTEXTE:\n{json.dumps(context, ensure_ascii=False)}\n\n"
            f"OPTIONS:\n{json.dumps([{'provider':e['provider'],'snippet':e['output'][:1500]} for e in entries], ensure_ascii=False)}\n\n"
            "TÂCHES:\n"
            "1) Scorer chaque option (rigor,usefulness,creativity,risk,faisabilite,impact_eco,roi).\n"
            "2) Sélectionner la meilleure selon critères & poids (risk pénalise), citer 2 forces / 1 faiblesse.\n"
            "3) Expliquer le choix (XAI court).\n"
            "4) Proposer un plan d'action (5-10 étapes) pour exécuter la décision.\n"
            "5) Lister 2-3 alternatives.\n"
            "6) Si ambiguïtés ou données insuffisantes, mettre raffinement_needed=true.\n"
        )
        agent_results[f"agent_{i}"] = _llm_json_call(prompt_agent, "Return ONLY JSON. No markdown.", temp=0.5)

    # Coordination / synthèse
    try:
        coord_prompt = (
            "Tu es le coordinateur final. Tu reçois les sorties de plusieurs agents spécialisés.\n"
            "Ta mission: fusionner et FINALISER une décision unique.\n"
            "Retourne UNIQUEMENT un JSON compact avec clés: choix, explication, plan (5-10 étapes), alternatives (2-3 items), xai_notes (liste courte).\n\n"
            f"AGENTS:\n{json.dumps(agent_results, ensure_ascii=False)}\n"
        )
        final_decision = _llm_json_call(coord_prompt, "Return ONLY JSON. No markdown.", temp=0.3)
    except Exception:
        # Fallback heuristique si échec
        scores = [heuristic_scores(e['output']) for e in entries]
        idx = scores.index(max(scores, key=lambda s: compute_total(s, weights)))
        final_decision = {
            "choix": entries[idx]['provider'],
            "explication": "Fallback heuristique: meilleur score pondéré.",
            "plan": ["1) Reprendre la sortie gagnante et la structurer", "2) Définir KPI et jalons", "3) Implémenter MVP", "4) Mesurer & itérer"],
            "alternatives": [e['provider'] for i,e in enumerate(entries) if i != idx],
            "xai_notes": ["Décision basée sur motifs de rigueur/utilité détectés."]
        }

    # Boucle de raffinement si demandé par agents
    try:
        needs_refine = any(ar.get("raffinement_needed") for ar in agent_results.values() if isinstance(ar, dict))
    except Exception:
        needs_refine = False

    loop = 0
    while needs_refine and loop < max_loops:
        loop += 1
        refine_prompt = (
            "Certains agents signalent un besoin de raffinement. Résous ambiguïtés et améliore le plan.\n"
            "Retourne UNIQUEMENT un JSON compact: choix, explication, plan (5-10 étapes), alternatives, xai_notes.\n\n"
            f"AGENTS:\n{json.dumps(agent_results, ensure_ascii=False)}\n"
            f"DECISION_ACTUELLE:\n{json.dumps(final_decision, ensure_ascii=False)}\n"
        )
        try:
            final_decision = _llm_json_call(refine_prompt, "Return ONLY JSON. No markdown.", temp=0.2)
            needs_refine = False
        except Exception:
            break

    # Persistance dans la mémoire projet
    if project_mem is not None:
        hist = project_mem.get("decisions_history", [])
        hist.append({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "criteria": user_criteria,
            "decision": final_decision
        })
        project_mem["decisions_history"] = hist

    return final_decision

# ======================== Onglet 1 — Orchestrateur IA ========================
with tab_orch:
    # ------- Mémoire projet -------
    st.subheader("🧠 Mémoire projet")
    colm1, colm2 = st.columns(2)
    with colm1:
        st.session_state["project_mem"]["brief"] = st.text_area("Brief produit", st.session_state["project_mem"]["brief"], height=80, key="mem_brief")
        st.session_state["project_mem"]["audience"] = st.text_area("Audience & ICP", st.session_state["project_mem"]["audience"], height=60, key="mem_aud")
        st.session_state["project_mem"]["voice"] = st.text_area("Ton de marque", st.session_state["project_mem"]["voice"], height=60, key="mem_voice")
    with colm2:
        st.session_state["project_mem"]["objectives"] = st.text_area("Objectifs (OKR)", st.session_state["project_mem"]["objectives"], height=60, key="mem_obj")
        st.session_state["project_mem"]["constraints"] = st.text_area("Contraintes (tech/brand/legal)", st.session_state["project_mem"]["constraints"], height=60, key="mem_cons")
        st.session_state["inject_mem"] = st.checkbox("🔗 Injecter la mémoire dans les prompts", value=st.session_state["inject_mem"], key="inject_mem_cb")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Exporter mémoire (.json)", data=json.dumps(st.session_state["project_mem"], ensure_ascii=False, indent=2),
                            file_name="ecoswitch_memory.json", mime="application/json", key="dl_mem")
    with c2:
        mem_upload = st.file_uploader("Charger mémoire (.json)", type=["json"], key="up_mem")
        if mem_upload:
            try:
                st.session_state["project_mem"] = json.load(mem_upload)
                st.success("Mémoire chargée.")
            except Exception as e:
                st.error(f"Erreur de chargement mémoire: {e}")

    # ------- RAG -------
    st.subheader("📚 Docs projet (RAG)")
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

    rag_row = st.columns(3)
    with rag_row[0]:
        st.session_state["use_rag"] = st.checkbox("Activer RAG", value=st.session_state["use_rag"], key="use_rag_cb")
    with rag_row[1]:
        st.session_state["rag_k"]  = st.slider("Nb d'extraits (top-K)", 1, 10, st.session_state["rag_k"], key="rag_k_sl")

    # ------- Paramètres IA -------
    st.subheader("⚙️ Paramètres IA")
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        st.session_state["use_gpt"]    = st.checkbox("GPT (OpenAI)", value=st.session_state["use_gpt"], key="use_gpt_cb")
        st.session_state["gpt_model"]  = st.text_input("GPT model", value=st.session_state["gpt_model"], key="gpt_model_in")
    with pcol2:
        st.session_state["use_grok"]   = st.checkbox("Grok (xAI)", value=st.session_state["use_grok"], key="use_grok_cb")
        st.session_state["grok_model"] = st.text_input("Grok model", value=st.session_state["grok_model"], key="grok_model_in")
    with pcol3:
        st.session_state["use_gemini"] = st.checkbox("Gemini (Google)", value=st.session_state["use_gemini"], key="use_gemini_cb")
        st.session_state["gemini_model"]= st.text_input("Gemini model", value=st.session_state["gemini_model"], key="gemini_model_in")

    pcol4, pcol5, pcol6 = st.columns(3)
    with pcol4:
        st.session_state["temperature"]   = st.slider("Température", 0.0, 1.5, st.session_state["temperature"], 0.1, key="temp_sl")
    with pcol5:
        st.session_state["debate_rounds"] = st.number_input("Débat — tours", 0, 5, st.session_state["debate_rounds"], 1, key="debate_nb")
    with pcol6:
        st.session_state["debate_style"]  = st.selectbox("Style de débat", ["coopératif","critique","agressif"],
                                            index=["coopératif","critique","agressif"].index(st.session_state["debate_style"]), key="debate_style_sel")
    st.session_state["max_chars"] = st.number_input("Max chars/échange (0=illimité)", 0, 20000, st.session_state["max_chars"], 500, key="max_chars_nb")

    st.subheader("🧮 Juge & Poids")
    j1, j2, j3, j4, j5 = st.columns(5)
    with j1:
        st.session_state["judge_kind"] = st.selectbox("Type", ["llm","heuristic"], index=["llm","heuristic"].index(st.session_state["judge_kind"]), key="judge_kind_sel")
    with j2:
        st.session_state["judge_provider"] = st.selectbox("Fournisseur juge (si llm)", ["gpt","grok","gemini"],
                                    index=["gpt","grok","gemini"].index(st.session_state["judge_provider"]), key="judge_provider_sel")
    with j3:
        st.session_state["judge_model"] = st.text_input("Modèle juge (optionnel)", value=st.session_state["judge_model"], key="judge_model_in")
    with j4:
        st.session_state["w_rigor"] = st.slider("Rigueur", 0, 100, st.session_state["w_rigor"], 1, key="w_rigor_sl")
    with j5:
        st.session_state["w_use"]   = st.slider("Utilité", 0, 100, st.session_state["w_use"], 1, key="w_use_sl")
    j6, j7 = st.columns(2)
    with j6:
        st.session_state["w_crea"]  = st.slider("Créativité", 0, 100, st.session_state["w_crea"], 1, key="w_crea_sl")
    with j7:
        st.session_state["w_risk"]  = st.slider("Risque (pénalise)", 0, 100, st.session_state["w_risk"], 1, key="w_risk_sl")
    weights = normalize_weights(st.session_state["w_rigor"], st.session_state["w_use"], st.session_state["w_crea"], st.session_state["w_risk"])
    st.caption(f"Poids → R:{weights['rigor']:.2f} U:{weights['usefulness']:.2f} C:{weights['creativity']:.2f} K:{weights['risk']:.2f}")

    # ------- Playbooks -------
    st.subheader("📒 Playbooks ECOSWITCH")
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

    # ------- Prompt principal -------
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
            ctx, _ = build_rag_context(user_prompt, st.session_state["kb_store"], top_k=st.session_state["rag_k"])
            if ctx.strip():
                blocks.append("### EXTRAITS DOCS (RAG)\n" + ctx)
        blocks.append("### TÂCHE\n" + (user_prompt or ""))
        return "\n\n".join([b for b in blocks if b.strip()])

    prompt = st.text_area("🧠 Prompt", height=140, key="prompt", placeholder="Décris la tâche ECOSWITCH…")
    system = st.text_input("🗣️ System prompt (optionnel)", key="system")
    colA, colB, colC = st.columns([1,1,1])
    with colA: run_btn = st.button("🚀 Lancer", type="primary", key="run_btn")
    with colB: clear_btn = st.button("🧽 Nettoyer", key="clear_btn")
    with colC: save_local = st.checkbox("Activer exports (JSON/MD/HTML)", value=True, key="save_local_cb")

    if clear_btn:
        st.session_state["prompt"] = ""
        st.session_state["system"] = "Tu es un comité d'experts (ingénierie, UX, marché). Style: clair, structuré, actionnable."
        st.stop()

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
        ordered = []
        for name in ["gpt","grok","gemini"]:
            for r in calls:
                if r.provider == name and r.ok:
                    ordered.append(r)
        return ordered

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

    # ===== RUN =====
    if run_btn:
        if not (prompt or "").strip():
            st.error("Merci d'écrire un prompt.")
        else:
            final_prompt = assemble_task_prompt(prompt)
            rag_ctx_current = get_rag_ctx_for_prompt(prompt)  # pour Décideur Auto
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
                weights_local = normalize_weights(st.session_state["w_rigor"], st.session_state["w_use"], st.session_state["w_crea"], st.session_state["w_risk"])
                scoreboard = judge_with_provider(prompt, entries, provider_for_judge, st.session_state["judge_model"] or None, weights_local)

                # --- Décideur Auto (auto-run après scoring) ---
                try:
                    st.session_state["last_decision"] = decideur_auto(
                        entries, weights_local, st.session_state["criteria_text"],
                        project_mem=st.session_state["project_mem"], rag_ctx=rag_ctx_current,
                        max_agents=3, max_loops=2
                    )
                except Exception as e:
                    st.session_state["last_decision"] = {"erreur": str(e)}

            st.success("Terminé !")

    # ------- Affichage résultats -------
    if entries:
        tabs_out = st.tabs([f"{e['provider'].upper()}" for e in entries] + ["📊 Scores"] + (["💬 Débat"] if transcript else []) + (["📤 Export"]))

        # sorties modèles
        for i, e in enumerate(entries):
            with tabs_out[i]:
                st.caption(f"Modèle: {e['model']} • Latence: {e.get('latency_s',0):.2f}s")
                st.text_area(f"Sortie – {e['provider'].upper()}", value=e["output"], height=350, key=f"out_{i}_{e['provider']}")

        # scores + DECIDEUR
        with tabs_out[len(entries)]:
            rows = []
            weights_local = normalize_weights(st.session_state["w_rigor"], st.session_state["w_use"], st.session_state["w_crea"], st.session_state["w_risk"])
            for s in scoreboard.get("scores", []):
                rows.append({
                    "provider": s["provider"],
                    "rigor": round(s["rigor"],2),
                    "usefulness": round(s["usefulness"],2),
                    "creativity": round(s["creativity"],2),
                    "risk": round(s["risk"],2),
                    "total": round(compute_total(s, weights_local),2)
                })
            if rows:
                st.dataframe(rows, use_container_width=True)
                ranking = " > ".join(scoreboard.get("weighted_ranking", [])) or "(n/a)"
                st.markdown(f"**Classement pondéré :** {ranking}")
            st.subheader("Synthèse"); st.write(scoreboard.get("final_synthesis","(n/a)"))
            st.subheader("Plan d'action"); st.write(scoreboard.get("action_plan","(n/a)"))

            st.divider()
            st.subheader("🧠 Décideur Auto (multi-agents, XAI)")
            st.session_state["criteria_text"] = st.text_input("Critères de décision",
                value=st.session_state["criteria_text"], key="criteria_text_in")
            colD1, colD2 = st.columns([1,1])
            with colD1:
                if st.button("🔎 Prendre une décision (recalculer)", key="decide_btn"):
                    try:
                        rag_ctx_current = get_rag_ctx_for_prompt(prompt)
                        st.session_state["last_decision"] = decideur_auto(
                            entries, weights_local, st.session_state["criteria_text"],
                            project_mem=st.session_state["project_mem"], rag_ctx=rag_ctx_current,
                            max_agents=3, max_loops=2
                        )
                    except Exception as e:
                        st.session_state["last_decision"] = {"erreur": str(e)}
            with colD2:
                if st.button("🔁 Raffiner (plan-execute +1)", key="refine_btn"):
                    try:
                        rag_ctx_current = get_rag_ctx_for_prompt(prompt)
                        st.session_state["last_decision"] = decideur_auto(
                            entries, weights_local, st.session_state["criteria_text"] + " (raffiné)",
                            project_mem=st.session_state["project_mem"], rag_ctx=rag_ctx_current,
                            max_agents=3, max_loops=3
                        )
                    except Exception as e:
                        st.session_state["last_decision"] = {"erreur": str(e)}

            if st.session_state["last_decision"]:
                with st.expander("Voir la décision détaillée", expanded=True):
                    st.json(st.session_state["last_decision"])

        # Débat
        if transcript:
            with tabs_out[len(entries)+1]:
                st.write("Transcription du débat (extraits)")
                for rd in sorted(set(t["round"] for t in transcript)):
                    st.markdown(f"#### Round {rd}")
                    for t in [x for x in transcript if x["round"] == rd]:
                        with st.expander(f"Round {rd} — {t['speaker'].upper()} ({t['model']})"):
                            st.text(t["text"])

        # Export
        with tabs_out[-1]:
            kb = st.session_state.get("kb_store", {})
            kb_info = {"files": kb.get("files", []), "chunks_count": len(kb.get("chunks", []))}
            meta_pack = {
                "gpt_model": st.session_state["gpt_model"], "grok_model": st.session_state["grok_model"], "gemini_model": st.session_state["gemini_model"],
                "active": [p for p,flag in (("gpt",st.session_state["use_gpt"]),("grok",st.session_state["use_grok"]),("gemini",st.session_state["use_gemini"])) if flag],
                "temp": st.session_state["temperature"], "debate_rounds": st.session_state["debate_rounds"], "debate_style": st.session_state["debate_style"], "max_chars": st.session_state["max_chars"]
            }
            bundle = {"prompt": prompt, "system": system, "entries": entries, "scoreboard": scoreboard, "meta": meta_pack,
                      "weights": normalize_weights(st.session_state["w_rigor"], st.session_state["w_use"], st.session_state["w_crea"], st.session_state["w_risk"]),
                      "kb": kb_info}
            if transcript: bundle["transcript"] = transcript
            md = build_markdown_report(prompt, system, entries, scoreboard, meta_pack, transcript,
                                       normalize_weights(st.session_state["w_rigor"], st.session_state["w_use"], st.session_state["w_crea"], st.session_state["w_risk"]),
                                       kb_info)
            html = build_html_report(md, title="Rapport Team IA – ECOSWITCH")
            st.download_button("⬇️ JSON (résultats)", data=json.dumps(bundle, ensure_ascii=False, indent=2),
                               file_name="results.json", mime="application/json", key="dl_json")
            st.download_button("⬇️ Markdown (.md)", data=md, file_name="rapport.md",
                               mime="text/markdown", key="dl_md")
            st.download_button("⬇️ HTML (.html)", data=html, file_name="rapport.html",
                               mime="text/html", key="dl_html")

# ======================= Onglet 2 — Simulation énergétique ====================
with tab_sim:
    st.header("🔬 Simulateur énergétique (bâtiment — modèle simple)")
    st.caption("But: illustrer l’impact de paramètres (surface, U, setpoint) sur la demande thermique quotidienne. Approximations pédagogiques.")

    colS1, colS2, colS3 = st.columns(3)
    with colS1:
        A = st.number_input("Surface (m²)", 10.0, 20000.0, 500.0, 10.0, key="sim_area")
        U = st.number_input("Coefficient global U (W/m²·K)", 0.1, 3.0, 0.8, 0.05, key="sim_u")
    with colS2:
        Ti_day = st.slider("Consigne intérieure jour (°C)", 16, 25, 21, 1, key="sim_ti_day")
        Ti_night = st.slider("Consigne intérieure nuit (°C)", 12, 21, 18, 1, key="sim_ti_night")
    with colS3:
        ext_base = st.selectbox("Profil extérieur", ["Hiver doux (Paris)", "Hiver froid (Lille)", "Mi-saison", "Été"], index=0, key="sim_ext")
        hvac_hours = st.slider("Heures HVAC (h/jour)", 0, 24, 16, 1, key="sim_hvac")

    if ext_base == "Hiver doux (Paris)":
        Te = np.array([2,2,1,1,1,2,3,4,6,7,8,9,9,8,7,5,4,3,3,3,3,3,3,3], dtype=float)
    elif ext_base == "Hiver froid (Lille)":
        Te = np.array([-2,-2,-3,-3,-2,-1,0,1,2,3,3,4,4,3,2,1,0,-1,-1,-2,-2,-2,-1,-1], dtype=float)
    elif ext_base == "Mi-saison":
        Te = np.array([8,8,7,7,7,8,10,12,14,15,16,17,18,17,16,14,12,10,9,9,9,9,8,8], dtype=float)
    else:
        Te = np.array([18,18,17,17,17,18,20,22,24,26,28,30,31,30,29,27,25,23,22,21,20,20,19,19], dtype=float)

    schedule = np.array([Ti_day if 8 <= h < (8+st.session_state.get("sim_hvac",16)) else Ti_night for h in range(24)], dtype=float)
    deltaT = np.maximum(0.0, schedule - Te)
    P = U * A * deltaT
    E_kWh = P / 1000.0
    total_kWh = float(np.sum(E_kWh))

    st.write(f"**Demande quotidienne estimée : {total_kWh:.1f} kWh** (chauffage)")

    fig = plt.figure(figsize=(7,3))
    plt.plot(range(24), E_kWh, marker="o")
    plt.title("Demande horaire (kWh)")
    plt.xlabel("Heure"); plt.ylabel("kWh/h")
    st.pyplot(fig, clear_figure=True)

    Ti_night_opt = max(12, Ti_night-2)
    hvac_opt = max(0, st.session_state.get("sim_hvac",16)-1)
    schedule_opt = np.array([Ti_day if 8 <= h < (8+hvac_opt) else Ti_night_opt for h in range(24)], dtype=float)
    E_opt = (U*A*np.maximum(0.0, schedule_opt-Te))/1000.0
    total_opt = float(np.sum(E_opt))
    gain = total_kWh - total_opt
    st.info(f"Scénario optimisé: **{total_opt:.1f} kWh**  → **gain ≈ {gain:.1f} kWh/jour**")

    fig2 = plt.figure(figsize=(7,3))
    plt.plot(range(24), E_kWh, label="Base", marker="o")
    plt.plot(range(24), E_opt, label="Optimisé", marker="s")
    plt.legend(); plt.title("Comparaison Base vs Optimisé"); plt.xlabel("Heure"); plt.ylabel("kWh/h")
    st.pyplot(fig2, clear_figure=True)

    st.caption("Modèle simplifié; à raffiner (HVAC réel, apports internes/solaires, inertie, COP, etc.).")

# =================== Onglet 3 — Générateur de code & tests ====================
with tab_code:
    st.header("🧪 Générateur de code & tests (IA)")
    st.caption("Décris la feature; l’IA propose un fichier de code + un fichier de tests. Tu peux sauvegarder, et exécuter localement (optionnel).")

    colC1, colC2 = st.columns([2,1])
    with colC1:
        spec = st.text_area("Spécification / Demande", height=160, key="code_spec",
            placeholder="Ex: Écris un endpoint FastAPI /ingest qui accepte JSON {meter_id, ts, kwh} et stocke dans SQLite; ajoute validation et test Pytest.")
    with colC2:
        framework = st.selectbox("Cible", ["FastAPI (Python)","Flask (Python)","Node/Express (JS)"], index=0, key="code_framework")
        want_tests = st.checkbox("Inclure tests unitaires", value=True, key="code_tests")
        run_after = st.checkbox("⚠️ Exécuter après sauvegarde (local)", value=False, key="code_run")

    sys_prompt_code = (
        "You are a senior full-stack engineer. Write production-grade code, with comments and clear structure. "
        "Prefer simplicity and readability. If tests requested, include realistic unit tests. "
        "Return ONLY code blocks for main file and tests file, with filenames on first line as comments."
    )
    if st.button("Générer le code", key="code_gen_btn"):
        model_for_code = st.session_state["gpt_model"]
        target_desc = f"Target framework: {framework}. Include tests: {want_tests}."
        user_prompt = f"{target_desc}\n\nSPEC:\n{spec}"
        r = ask_openai_gpt(user_prompt, sys_prompt_code, model_for_code, 0.3)
        if not r.ok:
            st.error(f"Erreur génération: {r.error}")
        else:
            txt = r.output
            blocks = re.findall(r"```(?:\w+)?\n(.*?)```", txt, flags=re.S)
            if not blocks:
                st.warning("Pas de bloc de code détecté; affichage brut:")
                st.text_area("Code brut", value=txt, height=400, key="code_raw")
            else:
                main_code = blocks[0]
                tests_code = blocks[1] if (want_tests and len(blocks)>1) else ""
                st.text_area("📄 Fichier principal", value=main_code, height=320, key="code_main_out")
                if tests_code:
                    st.text_area("🧪 Fichier de tests", value=tests_code, height=260, key="code_tests_out")

                col_save = st.columns(3)
                with col_save[0]:
                    fname_main = st.text_input("Nom fichier principal", "app_generated.py", key="code_fname_main")
                with col_save[1]:
                    fname_tests = st.text_input("Nom fichier tests", "test_generated.py", key="code_fname_tests")
                with col_save[2]:
                    base_dir = st.text_input("Dossier de sortie", "generated_code", key="code_out_dir")

                if st.button("💾 Sauvegarder fichiers", key="code_save_btn2"):
                    os.makedirs(base_dir, exist_ok=True)
                    if main_code:
                        with open(os.path.join(base_dir, fname_main), "w", encoding="utf-8") as f: f.write(main_code)
                    if tests_code:
                        with open(os.path.join(base_dir, fname_tests), "w", encoding="utf-8") as f: f.write(tests_code)
                    st.success(f"Fichiers écrits dans ./{base_dir}")

                    if run_after and fname_main.endswith(".py"):
                        try:
                            res = subprocess.run([sys.executable, os.path.join(base_dir, fname_main)], capture_output=True, text=True, timeout=20)
                            st.text_area("🖥️ STDOUT", value=res.stdout, height=160, key="code_run_stdout")
                            if res.stderr.strip():
                                st.text_area("⚠️ STDERR", value=res.stderr, height=160, key="code_run_stderr")
                            else:
                                st.success("Exécution terminée (voir STDOUT).")
                        except Exception as e:
                            st.error(f"Exécution échouée: {e}")

# ======================= Onglet 4 — Analyse marché ============================
with tab_market:
    st.header("📈 Analyse de marché / concurrents (light)")
    st.caption("Colle quelques URLs publiques (sites produits, pricing, docs). L’app récupère le texte, puis l’IA synthétise un tableau comparatif + SWOT.")

    urls_text = st.text_area("URLs (une par ligne)", height=120, key="market_urls",
        placeholder="https://exemple-produit-1.com\nhttps://exemple-produit-2.com/pricing\n...")
    query_focus = st.text_input("Focus (features, pricing, ICP, différenciation…)", "features & pricing & ICP", key="market_focus")
    fetch_btn = st.button("Analyser", key="market_analyze_btn")

    if fetch_btn:
        urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
        if not urls:
            st.error("Ajoute au moins une URL.")
        elif not REQUESTS_OK:
            st.error("Modules requests/bs4 manquants; ajoute-les au requirements.")
        else:
            with st.spinner("Récupération & synthèse…"):
                pages = []
                for u in urls:
                    try:
                        r = requests.get(u, timeout=15, headers={"User-Agent":"Mozilla/5.0"})
                        soup = BeautifulSoup(r.text, "html.parser")
                        for tag in soup(["script","style","noscript"]): tag.extract()
                        text = " ".join(soup.get_text(separator=" ").split())
                        pages.append({"url": u, "text": text[:30000]})
                    except Exception as e:
                        pages.append({"url": u, "text": f"[ERREUR FETCH: {e}]"})
                corpus = "\n\n".join([f"URL: {p['url']}\nTEXT:\n{p['text']}" for p in pages])
                sys_market = "You are a B2B market analyst. Return concise, actionable, structured outputs with tables in Markdown."
                usr_market = f"Focus: {query_focus}\nCompare les URLs suivantes. Livrables: 1) tableau Features x Produits; 2) pricing (si dispo); 3) ICP/segments; 4) SWOT synthétique; 5) 3 opportunités pour EcoSwitch.\n\n{corpus}"
                r = ask_openai_gpt(usr_market, sys_market, st.session_state["gpt_model"], 0.4)
                if r.ok:
                    st.markdown(r.output)
                else:
                    st.error(f"Erreur IA: {r.error}")

# ==================== Onglet 5 — PM / Kanban (data editor) ====================
with tab_pm:
    st.header("📋 Project Management (Kanban léger)")
    st.caption("Édite les tâches, génère une roadmap via IA, exporte/importe en JSON.")

    if "pm_df" not in st.session_state:
        st.session_state["pm_df"] = pd.DataFrame([
            {"id":1,"title":"Setup orchestrateur","owner":"Saadi","status":"Done","estimate_days":1,"due_date":""},
            {"id":2,"title":"PoC capteur IoT","owner":"Saadi","status":"Doing","estimate_days":3,"due_date":""},
            {"id":3,"title":"Value props v1","owner":"Saadi","status":"Todo","estimate_days":2,"due_date":""},
        ])

    colPM1, colPM2 = st.columns([2,1])
    with colPM1:
        st.session_state["pm_df"] = st.data_editor(
            st.session_state["pm_df"],
            num_rows="dynamic",
            use_container_width=True,
            key="pm_editor"
        )
    with colPM2:
        pm_goal = st.text_area("Objectif à planifier", height=140, key="pm_goal",
                               placeholder="Roadmap 6 mois pour MVP EcoSwitch (capteurs, dashboard, billing, conformité).")
        if st.button("Générer des tâches (IA)", key="pm_gen_btn"):
            sys_pm = "You are a pragmatic product manager. Return tasks as JSON list with fields: id, title, owner, status, estimate_days, due_date (YYYY-MM-DD or empty). Keep 8-15 tasks."
            r = ask_openai_gpt(pm_goal, sys_pm, st.session_state["gpt_model"], 0.5)
            if r.ok:
                try:
                    m = re.search(r"(\[.*\])", r.output, flags=re.S)
                    data = json.loads(m.group(1) if m else r.output)
                    df = pd.DataFrame(data)
                    for col in ["id","title","owner","status","estimate_days","due_date"]:
                        if col not in df.columns: df[col] = ""
                    st.session_state["pm_df"] = df[["id","title","owner","status","estimate_days","due_date"]]
                    st.success("Tâches générées.")
                except Exception as e:
                    st.error(f"Parsing JSON échoué: {e}")
            else:
                st.error(f"Erreur IA: {r.error}")

        st.download_button("⬇️ Exporter tâches (.json)", data=st.session_state["pm_df"].to_json(orient="records", force_ascii=False),
                           file_name="ecoswitch_tasks.json", mime="application/json", key="pm_dl")
        up_tasks = st.file_uploader("Importer tâches (.json)", type=["json"], key="pm_up")
        if up_tasks:
            try:
                st.session_state["pm_df"] = pd.DataFrame(json.load(up_tasks))
                st.success("Tâches importées.")
            except Exception as e:
                st.error(f"Import échoué: {e}")
