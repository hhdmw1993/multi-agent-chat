# 🎙️ Multi-Chat — AI Roundtable Discussion System

> Multi-Agent Discussion Room: Ask a question → Host creates agenda → Guests speak in real-time → Interactive Q&A → Summary Report

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![Vue](https://img.shields.io/badge/Vue_3-CDN-brightgreen?logo=vue.js)

---

## ✨ Key Features

- 🤖 **Multi-Model Support** — OpenAI / DeepSeek / Claude / Gemini / Ollama, unified OpenAI-compatible protocol
- 🎭 **Intelligent Host** — Four moderation styles (Neutral / Provocative / Guiding / Analytical)
- 📋 **Auto Agenda Generation** — Host auto-generates discussion topics based on your question
- 🗣️ **Personalized Guest Speech** — Role-based perspectives + web search for supporting materials
- 💬 **Audience Q&A** — Free-form questions after the agenda, host assigns guests to answer
- 📄 **Export Summary Report** — One-click download of meeting records
- 🔍 **Tavily Web Search** — Auto-search during guest prep, real-time fact-checking
- ⏸️ **Pause & Resume** — Pause anytime, continue later
- 💾 **History** — Auto-saved sessions, reviewable and reconfigurable

## 🖼️ Interface Preview

```
Home (Session Cards) → Setup (Select Roles + Topic) → Meeting (Message Stream + Interaction)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Modern browser (Chrome / Edge / Safari)

### Start Backend

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main_v2:app --host 0.0.0.0 --port 8001
```

### Open Frontend

Open `frontend/index.html` in your browser, or serve with any static file server.

### Configure Models

Add models via the top-right **"Model Config"** panel:
- **Platform**: e.g., DeepSeek, OpenAI, Ollama
- **Base URL**: up to `/v1` (e.g., `https://api.deepseek.com/v1`)
- **API Key**: Your platform's API key
- **Model Name**: e.g., `deepseek-chat`, `gpt-4o`

> For local Ollama models, no API key needed. Set Base URL to `http://localhost:11434/v1`

### Configure Web Search (Optional)

Enter your [Tavily API Key](https://tavily.com) (free tier available) under **"Smart Features" → "Web Search"**. When enabled, guests will automatically search for the latest information during preparation, and the host will also search on-demand when answering user questions.

### Configure Embedding Model (Optional)

Select an embedding platform under **"Smart Features" → "Vector Memory"**:

| Platform | Base URL | Model | Dimensions | Price |
|----------|----------|-------|------------|-------|
| Alibaba DashScope | `dashscope.aliyuncs.com/compatible-mode/v1` | `text-embedding-v4` | 1024 | Free 1M Tokens/month |
| OpenAI | `api.openai.com/v1` | `text-embedding-3-small` | 1536 | $0.02/1K tokens |
| Tencent HunYuan | `api.hunyuan.cloud.tencent.com/v1` | `hunyuan-embedding` | 1024 | Pay-as-you-go |
| Custom | Manual entry | Manual entry | — | — |

> With Embedding configured, guests can deeply understand other speakers' viewpoints and emotions during the discussion, enabling more targeted responses. Without it, the system falls back to an in-memory solution — all features work normally.

## 📁 Project Structure

```
multi-agent-chat/
├── backend/
│   ├── main_v2.py          # FastAPI entry point (port 8001)
│   ├── core/
│   │   ├── db.py           # SQLite persistence
│   │   ├── meeting_engine.py # Meeting engine (agenda/speech/Q&A)
│   │   └── model_adapter.py # Unified model adapter
│   │   ├── schemas.py      # Data models
│   │   └── prompts.py      # Prompt templates
│   └── utils/
│       ├── search_tool.py  # Tavily web search
│       └── vector_memory.py # Vector memory (ChromaDB)
├── frontend/
│   └── index.html          # Single-file SPA (Vue 3 CDN)
└── locales/
    ├── zh.json             # Chinese translations
    └── en.json             # English translations
```

## 🎮 How to Use

1. **Configure Models** — Add at least one AI model from the top-right panel
2. **New Meeting** — Choose host style → Add guests (name + stance) → Enter topic
3. **Confirm Agenda** — Host generates discussion agenda, you can request changes
4. **Guest Preparation** — Each guest thinks independently + searches the web
5. **Roundtable Discussion** — Guests speak turn-by-turn via streaming, pause/skip/call anytime
6. **Audience Q&A** — Free-form questions after agenda completes
7. **Download Report** — Export summary or full conversation log

## ⚙️ Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python FastAPI + SSE + WebSocket + SQLite |
| Frontend | Vue 3 (CDN) + Pure HTML/CSS |
| AI | Unified OpenAI-compatible interface, 10+ model providers |
| Search | Tavily API (optional) |
| Vector | ChromaDB (optional, with memory fallback) |

## 📜 License

MIT
