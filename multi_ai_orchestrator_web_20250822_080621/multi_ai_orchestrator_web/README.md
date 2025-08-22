# Multi‑IA Orchestrator — Web (Streamlit)

Version web *no‑PowerShell* : interface Streamlit, déployable sur **Streamlit Community Cloud** ou **Hugging Face Spaces**.

## Déploiement (Streamlit Cloud)
1) Crée un repo GitHub et pousse ce dossier.
2) Va sur https://share.streamlit.io → "New app" → choisis ton repo/branche.
3) Ajoute tes **secrets** (Settings → Secrets):
```
OPENAI_API_KEY = "sk-..."
XAI_API_KEY = "xai-..."
GOOGLE_API_KEY = "AIza..."
```
4) Lance l'app 🎉

## Déploiement (Hugging Face Spaces)
- Crée un Space (type **Streamlit**).
- Upload ce dossier.
- Définis 3 **secrets** dans "Variables": `OPENAI_API_KEY`, `XAI_API_KEY`, `GOOGLE_API_KEY`.

## Local (optionnel)
```bash
pip install -r requirements.txt
streamlit run app.py
```
