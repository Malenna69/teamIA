# -*- coding: utf-8 -*-
import os, json, time
import streamlit as st

# Providers
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Callable

# =============== Low-level provider calls ===============
from openai import OpenAI
from google import genai
from google.genai import types as genai_types

TIMEOUT_S = 60

def _get_key(name: str) -> Optional[str]:
    # priorities: st.secrets -> env -> text input
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
        res = client.models.generate_content(
            model=model,
            contents=prompt,
            config=cfg
        )
        txt = (res.text or "").strip()
        return ProviderResult("gemini", model, txt, time.time()-t0, True)
    except Exception as e:
        return ProviderResult("gemini", model, "", time.time()-t0, False, str(e))

# =============== Heuristic scoring ===============
import re
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

# =============== Judge via chosen provider ===============
def judge_with_provider(task: str, entries: list, provider: str, judge_model: Optional[str]):
    WEIGHTS = {"rigor":0.40,"usefulness":0.30,"creativity":0.20,"risk":0.10}
    if provider == "heuristic":
        scores = []
        for e in entries:
            s = heuristic_scores(e["output"]); s["provider"] = e["provider"]; scores.append(s)
        totals = [(s["provider"], 0.40*s["rigor"] + 0.30*s["usefulness"] + 0.20*s["creativity"] + 0.10*(10 - s["risk"])) for s in scores]
        ranking = [p for p,_ in sorted(totals, key=lambda x: x[1], reverse=True)]
        return {"scores": scores, "weighted_ranking": ranking,
                "final_synthesis": "Synthèse non‑LLM: utiliser le top‑rank comme base et fusionner manuellement.",
                "action_plan": "- Prendre les points clés du meilleur score\n- Ajouter 2 actions du 2e meilleur\n- Lister les risques signalés"}

    bundle = {"task": task, "entries": entries, "weights": WEIGHTS}
    prompt = (
        "You are an impartial evaluator.\n"
        "Return STRICT JSON only with keys: scores[], weighted_ranking[], final_synthesis, action_plan.\n"
        "Scoring schema: rigor/usefulness/creativity/risk (0..10). "
        "Compute weighted ranking (risk decreases total). "
        "Then produce final_synthesis and action_plan.\n\n"
        f"DATA:\n{json.dumps(bundle, ensure_ascii=False)}"
    )

    if provider == "gpt" or provider == "grok":
        base_url = "https://api.x.ai/v1" if provider=="grok" else None
        api_key = _get_key("XAI_API_KEY") if provider=="grok" else _get_key("OPENAI_API_KEY")
        if not api_key: raise RuntimeError("Clé API manquante pour le juge sélectionné.")
        client = OpenAI(api_key=api_key, base_url=base_url)
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

# =============== Debate mode ===============
def role_system(base: Optional[str]) -> str:
    base = base or "You are a respectful, concise expert."
    extra = (
        " You are in a multi‑agent debate. "
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
        "- Then provide a clear, improved REVISION (self‑contained)."
    )
    return "\n".join(blocks)

def run_debate(task: str, system: Optional[str], rounds: int,
               use_gpt: bool, use_grok: bool, use_gemini: bool,
               gpt_model: str, grok_model: str, gemini_model: str,
               temp: Optional[float]):
    participants = []
    if use_gpt: participants.append(("gpt", ask_openai_gpt, gpt_model))
    if use_grok: participants.append(("grok", ask_grok_xai, grok_model))
    if use_gemini: participants.append(("gemini", ask_gemini, gemini_model))

    drafts = {}
    transcript = []
    # Round 0: initial drafts
    for name, fn, model in participants:
        r = fn(task, role_system(system), model, temp)
        text = r.output if r.ok else f"[ERROR] {r.error}"
        drafts[name] = text
        transcript.append({"round": 0, "speaker": name, "model": model, "text": text})

    # Debate rounds
    for rd in range(1, rounds+1):
        new_drafts = {}
        for name, fn, model in participants:
            prompt = prompt_for_round(task, name, drafts, rd)
            r = fn(prompt, role_system(system), model, temp)
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

# =============== UI ===============
st.set_page_config(page_title="Multi‑IA Orchestrator (Web)", layout="wide")

st.title("🤝 Multi‑IA Orchestrator — Web")
st.caption("Grok + GPT + Gemini • brainstorming, débat multi‑tours, évaluation pondérée")

with st.sidebar:
    st.header("⚙️ Configuration")

    # API keys
    st.subheader("Clés API")
    openai_key = st.text_input("OPENAI_API_KEY", value=_get_key("OPENAI_API_KEY") or "", type="password")
    xai_key = st.text_input("XAI_API_KEY", value=_get_key("XAI_API_KEY") or "", type="password")
    google_key = st.text_input("GOOGLE_API_KEY", value=_get_key("GOOGLE_API_KEY") or "", type="password")
    if openai_key: os.environ["OPENAI_API_KEY"] = openai_key
    if xai_key: os.environ["XAI_API_KEY"] = xai_key
    if google_key: os.environ["GOOGLE_API_KEY"] = google_key

    st.subheader("Fournisseurs")
    use_gpt = st.checkbox("GPT (OpenAI)", value=True)
    use_grok = st.checkbox("Grok (xAI)", value=True)
    use_gemini = st.checkbox("Gemini (Google)", value=True)

    st.subheader("Modèles")
    gpt_model = st.text_input("GPT model", "gpt-4o-mini")
    grok_model = st.text_input("Grok model", "grok-4")
    gemini_model = st.text_input("Gemini model", "gemini-2.5-flash")

    st.subheader("Paramètres")
    temp = st.slider("Température", 0.0, 1.5, 0.6, 0.1)
    debate_rounds = st.number_input("Débat — nombre de tours", min_value=0, max_value=5, value=0, step=1)

    st.subheader("Juge")
    judge_kind = st.selectbox("Type", ["llm","heuristic"], index=0)
    judge_provider = st.selectbox("Fournisseur juge (si llm)", ["gpt","grok","gemini"], index=0)
    judge_model = st.text_input("Modèle juge (optionnel)", "")

prompt = st.text_area("🧠 Prompt", height=160, placeholder="Décris ton besoin...")
system = st.text_input("🗣️ System prompt (optionnel)", value="Tu es un comité d'experts (ingénierie, UX, marché). Style: clair, structuré, actionnable.")

col1, col2 = st.columns([1,1])
with col1:
    run_btn = st.button("🚀 Lancer")
with col2:
    save_btn = st.checkbox("Sauvegarde locale (rapport JSON téléchargeable)", value=True)

if run_btn:
    if not prompt.strip():
        st.error("Merci d'écrire un prompt.")
    else:
        with st.spinner("Exécution en cours..."):
            # debate or single pass
            if debate_rounds and debate_rounds > 0:
                final_drafts, transcript = run_debate(
                    prompt, system, debate_rounds,
                    use_gpt, use_grok, use_gemini,
                    gpt_model, grok_model, gemini_model,
                    temp
                )
                # Convert drafts to entries for judging
                entries = []
                for provider, draft in final_drafts.items():
                    entries.append({
                        "provider": provider,
                        "model": {"gpt": gpt_model, "grok": grok_model, "gemini": gemini_model}[provider],
                        "latency_s": 0.0,
                        "ok": True,
                        "error": None,
                        "output": draft[:15000]
                    })
            else:
                # one-shot parallel: sequential in UI for simplicity
                entries = []
                results = []
                if use_gpt:
                    res = ask_openai_gpt(prompt, system, gpt_model, temp); results.append(res)
                if use_grok:
                    res = ask_grok_xai(prompt, system, grok_model, temp); results.append(res)
                if use_gemini:
                    res = ask_gemini(prompt, system, gemini_model, temp); results.append(res)
                ok = [r for r in results if r.ok]
                if not ok:
                    st.error("Aucune réponse valide — vérifie tes clés API et tes modèles.")
                    st.stop()
                for r in ok:
                    entries.append({
                        "provider": r.provider, "model": r.model, "latency_s": r.latency_s,
                        "ok": r.ok, "error": r.error, "output": r.output[:15000]
                    })
                transcript = None

            # judge
            provider_for_judge = judge_provider if judge_kind == "llm" else "heuristic"
            scoreboard = judge_with_provider(prompt, entries, provider_for_judge, judge_model or None)

        # Display results
        st.success("Terminé !")
        tabs = st.tabs([f"{e['provider'].upper()}" for e in entries] + ["📊 Scores"] + (["💬 Débat"] if transcript else []))

        for i, e in enumerate(entries):
            with tabs[i]:
                st.caption(f"Modèle: {e['model']} • Latence: {e['latency_s']:.2f}s")
                st.text_area("Sortie", value=e["output"], height=350)

        with tabs[len(entries)]:
            # Score table
            import pandas as pd
            rows = []
            for s in scoreboard.get("scores", []):
                total = 0.40*s["rigor"] + 0.30*s["usefulness"] + 0.20*s["creativity"] + 0.10*(10 - s["risk"])
                rows.append({"provider":s["provider"], "rigor":s["rigor"], "usefulness":s["usefulness"],
                             "creativity":s["creativity"], "risk":s["risk"], "total":round(total,2)})
            if rows:
                df = pd.DataFrame(rows).set_index("provider")
                st.dataframe(df)
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
            # provide downloadable json bundle
            bundle = {"prompt": prompt, "system": system, "entries": entries, "scoreboard": scoreboard}
            if transcript: bundle["transcript"] = transcript
            st.download_button("⬇️ Télécharger les résultats (JSON)", data=json.dumps(bundle, ensure_ascii=False, indent=2), file_name="results.json", mime="application/json")
