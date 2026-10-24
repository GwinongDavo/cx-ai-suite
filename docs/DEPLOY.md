# Deploying the CX AI Suite

About 30–45 minutes the first time. Everything here is free.

You'll end up with 4 live URLs (the eval harness lives inside the assistant at `/evals` and in GitHub Actions):

| App | Render service | Example URL |
|-----|----------------|-------------|
| Support Knowledge Assistant (+ `/evals`) | `cx-support-assistant` | `https://cx-support-assistant.onrender.com` |
| Ticket Triage Agent | `cx-triage-agent` | `https://cx-triage-agent.onrender.com` |
| SLA & Escalation Risk Dashboard | `cx-sla-dashboard` | `https://cx-sla-dashboard.onrender.com` |
| Call & Email QA Analyzer | `cx-qa-analyzer` | `https://cx-qa-analyzer.onrender.com` |

Render adds a random suffix if a name is taken; use whatever URLs it shows you.

---

## Step 1: Get an AI key (5 min)

**Gemini (free, recommended to start):**
1. Go to https://aistudio.google.com/apikey and sign in with a Google account.
2. Click **Create API key** and copy it. Keep it private: never paste it into code or commit it.

**Azure OpenAI (optional, matches the job description best):**
1. In the Azure portal, create an **Azure OpenAI** resource (Azure for Students or a free trial gives credit).
2. Open it in **Azure AI Foundry** → **Deployments** → deploy a chat model (e.g. `gpt-4.1-mini`) and an embedding
   model (`text-embedding-3-small`). The deployment names you choose go in `CX_CHAT_MODEL` / `CX_EMBED_MODEL`.
3. Copy the **key** and **endpoint** from the resource's *Keys and Endpoint* page.
4. In Step 3 below, set `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `CX_CHAT_MODEL`, `CX_EMBED_MODEL`
   instead of `GEMINI_API_KEY`. The app picks Azure automatically when both Azure values are set.

## Step 2: Put the code on GitHub (10 min)

Make this repo **public** so recruiters can read the code (keep your portfolio site repo private if you like).

1. On github.com → **New repository** → name it `cx-ai-suite` → Public → **don't** add a README → Create.
2. Unzip `cx-ai-suite.zip` somewhere, e.g. `C:\Users\<you>\code\cx-ai-suite`.
3. In PowerShell, inside that folder:

```powershell
git init -b main
git add .
git commit -m "CX AI Suite: RAG assistant, evals CI, triage agent, SLA dashboard, QA analyzer"
git remote add origin https://github.com/<your-username>/cx-ai-suite.git
git push -u origin main
```

4. Open the repo's **Actions** tab: the CI run should go green in ~2 minutes (tests + eval gate).
   Click the run to see the eval report in the summary.

## Step 3: Deploy all four apps on Render (10 min + build time)

1. Sign up at https://render.com with **Sign in with GitHub**, and allow it to see the `cx-ai-suite` repo.
2. Dashboard → **New** → **Blueprint** → choose `cx-ai-suite`. Render reads `render.yaml` and lists 4 web services.
3. It asks for the values marked secret in the `cx-ai-keys` group:
   - `GEMINI_API_KEY`: your key from Step 1 (leave empty to deploy in demo mode)
   - `CX_REPO_URL`: `https://github.com/<your-username>/cx-ai-suite`
4. Click **Apply**. Each service builds for 3–6 minutes. When they show **Live**, open each URL.
5. Check the badge in the top corner of the assistant and triage pages: it should say `LLM: gemini:gemini-3.8-flash`
   rather than `demo mode`.

**Free plan notes:** services sleep after ~15 minutes without visitors and take 30–60 seconds to wake.
Before an interview or sending a link, open each URL a minute early. The free plan has 750 instance-hours a month
shared across services, so don't set up an uptime pinger for all four.

**Optional, faster dashboards:** the two Streamlit apps can also go on https://share.streamlit.io
(New app → repo `cx-ai-suite` → main file `apps/sla_dashboard/app.py`, then again for `apps/qa_analyzer/app.py`;
put the key under *Advanced settings → Secrets* as `GEMINI_API_KEY = "..."`).

## Step 4: Turn on live evals in CI (3 min)

1. GitHub repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:
   `GEMINI_API_KEY` = your key (or the Azure/OpenAI ones).
2. **Actions** → **CI** → **Run workflow** → tick *Also run evals against the real LLM* → Run.
3. This scores the real model, including LLM-as-judge faithfulness, and fails if quality drops below
   `evals/thresholds.json`. If the live handoff numbers are off, look at the threshold table in the report and set
   `CX_HANDOFF_THRESHOLD` in Render to the best value.

## Step 5: Add them to your portfolio

Send the four URLs back in the project thread and I'll add the five project cards (with screenshots) to your
portfolio's `data.py`. Then rebuild and deploy as usual:

```powershell
python build.py
vercel --prod
```

## If something goes wrong

- **Render build fails on `pip install`:** check the Python version in the logs; `.python-version` asks for 3.13.
- **Page says "demo mode" even with a key:** the env var name must be exactly `GEMINI_API_KEY`; redeploy after changing it
  (Render → service → Manual Deploy).
- **Gemini 429 errors:** free-tier rate limit. The apps fall back to demo logic for that request; wait a minute.
- **CI red on the eval gate:** open the run summary; the "Cases to look at" table lists which questions failed.
