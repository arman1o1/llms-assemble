# 🧩 LLMs-Assemble App

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange?style=for-the-badge&logo=gradio&logoColor=white)
![LLMs](https://img.shields.io/badge/LLMs-Multi--Provider-purple?style=for-the-badge)

A Gradio-based web application that lets you **query and compare responses from multiple LLM providers simultaneously**, all in a single interface.

Designed for **side-by-side evaluation, qualitative analysis, and prompt experimentation** across models.

---

## 🖼️ App Screenshot

![LLMs Assemble Meme](llm-assemble-meme.png)

---

## ✨ Features

* **Multi-LLM Comparison:** Query OpenAI, Anthropic, Google, Groq, and Perplexity in parallel.
* **Chairman Mode:** Enable models to critique, discuss, and respond to each other’s outputs.
* **Real-Time Streaming:** View responses as they arrive.
* **Secure by Default:** API keys are stored in-memory only.
* **Developer-Friendly:** Ideal for prompt testing and model evaluation workflows.

---

## 🛠️ Tech Stack

* **UI:** Gradio
* **Backend:** Python
* **LLM Providers:**
  * OpenAI
  * Anthropic
  * Google
  * Groq
  * Perplexity

---

## 📦 Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
````

---

## ▶️ Usage

Run the application locally:

```bash
python app.py
```

Once started, Gradio will provide a local URL:

```text
Running on local URL: http://127.0.0.1:7860
```

Open the link in your browser to access the app.

---

## 🔐 Environment Variables

You can provide API keys either via the UI or by creating a `.env` file.

```text
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
GROQ_API_KEY=gsk_...
PERPLEXITY_API_KEY=pplx-...
```

---

## 📝 Notes

* API keys entered via the UI are stored **in-memory only** and are cleared when the app restarts.
* For persistent configuration, use environment variables instead.

---

## 🚧 Project Status

This project is **experimental and under active development**.

It is inspired by Andrej Karpathy’s
[llm-council](https://github.com/karpathy/llm-council)

Expect rough edges and ongoing improvements.

---

## 📄 License

The code in this repository is licensed under the **MIT License**.
