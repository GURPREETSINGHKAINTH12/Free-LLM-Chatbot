# Free LLM Chatbot

A simple, modular chatbot built with **Python + FastAPI** and a **free Large Language Model (LLM)**. It supports **Ollama locally** or **Hugging Face inference**. No OpenAI, ChatGPT, Gemini, or paid APIs are required.

## Features

- **Prompt templating:** System message + user/assistant conversation structure
- **Conversation memory:** Keeps the last 5 messages for context
- **Temperature control:** Adjustable through the UI from `0` to `2`
- **Simple frontend:** HTML + JavaScript interface for sending messages and viewing replies
- **Local inference:** Run a model on your own computer using Ollama
- **Alternative backend:** Optional Hugging Face inference with an access token

## Project structure

```text
LLM-Prompt Project/
├── app.py              # FastAPI backend and LLM calls
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Chat interface
└── README.md           # Project documentation
```

## Setup

### 1. Create and activate a virtual environment

Open PowerShell in the project directory:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Choose your LLM backend

#### Option A: Ollama (recommended)

Ollama is free and runs the model locally. No API key is required.

1. Install [Ollama](https://ollama.com).
2. Download a model. This project was tested with **Llama 3.2 (3B)**:

   ```powershell
   ollama pull llama3.2:3b
   ```

3. Set the model name in PowerShell:

   ```powershell
   $env:OLLAMA_MODEL = "llama3.2:3b"
   ```

4. Confirm Ollama is available:

   ```powershell
   ollama list
   ```

#### Option B: Hugging Face

Hugging Face inference is an optional alternative and requires an access token.

1. Create a token at [Hugging Face Access Tokens](https://huggingface.co/settings/tokens).
2. Set the environment variables in PowerShell:

   ```powershell
   $env:USE_HF = "true"
   $env:HF_TOKEN = "your_token_here"
   ```

3. Optionally set the model:

   ```powershell
   $env:HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
   ```

Keep your token private. **Do not commit it to GitHub.**

### 3. Start the application

From the project root, run:

```powershell
python -m uvicorn app:app --reload --app-dir .
```

Open the chatbot in your browser:

**http://127.0.0.1:8000**

To stop the server, press `Ctrl + C` in PowerShell.

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the chatbot interface |
| GET | `/health` | Reports backend and model status |
| POST | `/api/chat` | Sends a message and returns a response |

### Example request

```json
{
  "message": "Explain what an API is in simple words.",
  "temperature": 0.7,
  "history": []
}
```

### Example response structure

```json
{
  "reply": "An API allows software systems to communicate with each other.",
  "model_used": "llama3.2:3b",
  "history": []
}
```

The actual response and conversation history will vary.

## Notes

- **Ollama:** Free local inference with no API key. The model must be downloaded before use.
- **Hugging Face:** Optional hosted inference. Requires a token and availability depends on the selected model and service limits.
- **Conversation memory:** Stored in memory during the application session. Restarting the server clears server-side session data.
- **Temperature:** Lower values generally produce more focused responses; higher values can produce more varied responses.
