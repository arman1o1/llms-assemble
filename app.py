import gradio as gr
import os
from typing import Dict, Optional, List
import json
import random
import asyncio 
import threading

# Try to import required packages
try:
    import openai
    from openai import OpenAI
    import google.generativeai as genai
    import anthropic
    from groq import Groq
    OPENAI_AVAILABLE = True
except ImportError:
    print("Some packages not installed. Please install with:")
    print("pip install openai google-generativeai anthropic groq python-dotenv")
    OPENAI_AVAILABLE = False

# Load environment variables if .env exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Configuration
# REMOVED: CONFIG_FILE = "llm_config.json"
SUPPORTED_PROVIDERS = ["OpenAI", "Anthropic", "Google", "Groq", "Perplexity"]

# Model lists as per instructions
OPENAI_MODELS = [
    "gpt-5.1",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
]

ANTHROPIC_MODELS = [
    "claude-opus-4.5",
    "claude-sonnet-4.5",
    "claude-haiku-4.5",
    "claude-opus-4.1",
    "claude-sonnet-4",
]

GOOGLE_MODELS = [
    "gemini-3-pro-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

GROQ_MODELS = [
    "grok-4-1-fast-reasoning",
    "grok-4-1-fast-non-reasoning",
    "grok-4",
    "grok-3",
    "grok-3-mini",
]

PERPLEXITY_MODELS = [
    "sonar",
    "sonar-pro",
    "sonar-reasoning",
    "sonar-reasoning-pro",
    "sonar-deep-research",
    "r1-1776"
]

class LLMClient:
    """Client for interacting with different LLM providers"""
    
    def __init__(self):
        self.config = self.load_config()
        self.clients = {}
        self.initialize_clients()
    
    def load_config(self) -> Dict:
        """Load configuration from environment variables only"""
        default_config = {
            "providers": {
                "OpenAI": {"api_key": "", "model": "gpt-4o-mini"},
                "Anthropic": {"api_key": "", "model": "claude-opus-4.5"},
                "Google": {"api_key": "", "model": "gemini-2.5-flash"},
                "Groq": {"api_key": "", "model": "grok-4"},
                "Perplexity": {"api_key": "", "model": "sonar-reasoning"}
            },
            "active_providers": SUPPORTED_PROVIDERS,
            "temperature": 0.7,
            "max_tokens": 1000,
            "chairman_provider": "OpenAI"
        }
        
        # Load API keys from environment variables only
        env_keys = {
            "OpenAI": os.getenv("OPENAI_API_KEY", ""),
            "Anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
            "Google": os.getenv("GOOGLE_API_KEY", ""),
            "Groq": os.getenv("GROQ_API_KEY", ""),
            "Perplexity": os.getenv("PERPLEXITY_API_KEY", "")
        }
        
        for provider, key in env_keys.items():
            if key:
                default_config["providers"][provider]["api_key"] = key
        
        return default_config
    
    # REMOVED: save_config method completely
    
    def initialize_clients(self):
        """Initialize clients for each provider"""
        if not OPENAI_AVAILABLE:
            return
            
        providers = self.config["providers"]
        
        # Initialize OpenAI client
        if providers["OpenAI"]["api_key"]:
            self.clients["OpenAI"] = OpenAI(api_key=providers["OpenAI"]["api_key"])
        
        # Initialize Anthropic client
        if providers["Anthropic"]["api_key"]:
            self.clients["Anthropic"] = anthropic.Anthropic(
                api_key=providers["Anthropic"]["api_key"]
            )
        
        # Initialize Google client
        if providers["Google"]["api_key"]:
            genai.configure(api_key=providers["Google"]["api_key"])
            self.clients["Google"] = genai
        
        # Initialize Groq client
        if providers["Groq"]["api_key"]:
            self.clients["Groq"] = Groq(api_key=providers["Groq"]["api_key"])
        
        # Initialize Perplexity client (using OpenAI client with Perplexity endpoint)
        if providers["Perplexity"]["api_key"]:
            self.clients["Perplexity"] = OpenAI(
                api_key=providers["Perplexity"]["api_key"],
                base_url="https://api.perplexity.ai"
            )
    
    def query_openai(self, prompt: str) -> str:
        """Query OpenAI models"""
        try:
            response = self.clients["OpenAI"].chat.completions.create(
                model=self.config["providers"]["OpenAI"]["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI Error: {str(e)}"
    
    def query_anthropic(self, prompt: str) -> str:
        """Query Anthropic Claude models"""
        try:
            response = self.clients["Anthropic"].messages.create(
                model=self.config["providers"]["Anthropic"]["model"],
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"Anthropic Error: {str(e)}"
    
    def query_google(self, prompt: str) -> str:
        """Query Google Gemini models"""
        try:
            model = self.clients["Google"].GenerativeModel(
                self.config["providers"]["Google"]["model"]
            )
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config["temperature"],
                    max_output_tokens=self.config["max_tokens"]
                )
            )
            return response.text
        except Exception as e:
            return f"Google Error: {str(e)}"
    
    def query_groq(self, prompt: str) -> str:
        """Query Groq models"""
        try:
            response = self.clients["Groq"].chat.completions.create(
                model=self.config["providers"]["Groq"]["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Groq Error: {str(e)}"
    
    def query_perplexity(self, prompt: str) -> str:
        """Query Perplexity models"""
        try:
            response = self.clients["Perplexity"].chat.completions.create(
                model=self.config["providers"]["Perplexity"]["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Perplexity Error: {str(e)}"
    
    def query_all(self, prompt: str) -> Dict[str, str]:
        """Query all active providers"""
        results = {}
        active_providers = self.config["active_providers"]
        
        for provider in active_providers:
            if provider not in self.clients:
                results[provider] = f"API key not configured for {provider}"
                continue
                
            if provider == "OpenAI":
                results[provider] = self.query_openai(prompt)
            elif provider == "Anthropic":
                results[provider] = self.query_anthropic(prompt)
            elif provider == "Google":
                results[provider] = self.query_google(prompt)
            elif provider == "Groq":
                results[provider] = self.query_groq(prompt)
            elif provider == "Perplexity":
                results[provider] = self.query_perplexity(prompt)
        
        return results
    
    def query_provider(self, provider: str, prompt: str) -> str:
        """Query a specific provider"""
        if provider not in self.clients:
            return f"API key not configured for {provider}"
            
        if provider == "OpenAI":
            return self.query_openai(prompt)
        elif provider == "Anthropic":
            return self.query_anthropic(prompt)
        elif provider == "Google":
            return self.query_google(prompt)
        elif provider == "Groq":
            return self.query_groq(prompt)
        elif provider == "Perplexity":
            return self.query_perplexity(prompt)
        return f"Unknown provider: {provider}"
    
    def update_api_key(self, provider: str, api_key: str, model: str = None):
        """Update API key for a provider - TEMPORARY IN-MEMORY ONLY"""
        if provider in self.config["providers"]:
            # Only store in memory during this session
            if api_key:
                self.config["providers"][provider]["api_key"] = api_key
            if model:
                self.config["providers"][provider]["model"] = model
            # Reinitialize client with new API key
            self.initialize_clients()
            return True
        return False
    
    def update_active_providers(self, active_providers: list):
        """Update which providers are active - TEMPORARY IN-MEMORY ONLY"""
        self.config["active_providers"] = active_providers
    
    def update_settings(self, temperature: float, max_tokens: int):
        """Update generation settings - TEMPORARY IN-MEMORY ONLY"""
        self.config["temperature"] = temperature
        self.config["max_tokens"] = max_tokens
    
    def set_chairman_provider(self, chairman_provider: str):
        """Set the chairman provider - TEMPORARY IN-MEMORY ONLY"""
        self.config["chairman_provider"] = chairman_provider

# Initialize the LLM client
llm_client = LLMClient()

def query_llms(prompt: str) -> Dict[str, str]:
    """Main function to query all LLMs"""
    if not OPENAI_AVAILABLE:
        return {provider: "Required packages not installed" for provider in llm_client.config["active_providers"]}
    
    if not prompt.strip():
        return {provider: "Please enter a prompt" for provider in llm_client.config["active_providers"]}
    
    return llm_client.query_all(prompt)

# NEW FUNCTION: Process query with progressive updates
def process_query_with_updates(prompt, temp, tokens, progress=gr.Progress()):
    """Process query and yield results as they complete"""
    if not OPENAI_AVAILABLE:
        yield ["Required packages not installed"] * len(SUPPORTED_PROVIDERS)
        return
    
    if not prompt.strip():
        yield ["Please enter a prompt"] * len(SUPPORTED_PROVIDERS)
        return
    
    llm_client.update_settings(temp, tokens)
    
    # Initialize results with placeholders
    results = {provider: "⏳ Processing..." for provider in SUPPORTED_PROVIDERS}
    yield [results.get(provider, "Not active") for provider in SUPPORTED_PROVIDERS]
    
    # Use threads to query providers concurrently
    import queue
    result_queue = queue.Queue()
    
    def query_and_store(provider):
        """Query provider and store result in queue"""
        try:
            result = llm_client.query_provider(provider, prompt)
            result_queue.put((provider, result))
        except Exception as e:
            result_queue.put((provider, f"Error: {str(e)}"))
    
    threads = []
    for provider in SUPPORTED_PROVIDERS:
        thread = threading.Thread(target=query_and_store, args=(provider,))
        threads.append(thread)
        thread.start()
    
    # Update results as threads complete
    completed = 0
    while completed < len(SUPPORTED_PROVIDERS):
        try:
            provider, result = result_queue.get(timeout=5)
            results[provider] = result
            completed += 1
            # Yield updated results
            yield [results.get(provider, "⏳ Processing...") for provider in SUPPORTED_PROVIDERS]
        except queue.Empty:
            continue
    
    # Final yield with all results
    yield [results.get(provider, "⏳ Processing...") for provider in SUPPORTED_PROVIDERS]

# NEW FUNCTION: Process chairman discussion with progressive updates
def process_chairman_discussion_with_updates(prompt, providers, chairman, temp, tokens, progress=gr.Progress()):
    """Process chairman discussion and yield results as they complete"""
    if not providers:
        yield "Please select at least one AI participant", "", ""
        return
    
    # Yield initial placeholders
    yield "⏳ Getting initial answers...", "⏳ Waiting for critiques...", "⏳ Waiting for chairman..."
    
    # Initialize anonymized_answers
    anonymized_answers = {}
    
    # Step 1: Get initial answers with progress updates
    initial_answers = {}
    initial_text = ""
    
    progress(0.1, desc="Getting initial answers")
    for i, provider in enumerate(providers):
        if provider in llm_client.clients:
            initial_answers[provider] = llm_client.query_provider(provider, prompt)
        else:
            initial_answers[provider] = f"API key not configured for {provider}"
        
        # Update initial answers text
        initial_text += f"## {provider}\n{initial_answers[provider]}\n\n{'='*60}\n\n"
        yield initial_text, "⏳ Waiting for critiques...", "⏳ Waiting for chairman..."
        progress(0.1 + (0.3 * (i + 1) / len(providers)), desc=f"Got answer from {provider}")
    
    # Step 2: Get critiques (if we have at least 2 providers)
    critiques_text = ""
    if len(providers) > 1:
        progress(0.4, desc="Getting critiques")
        # Create anonymized versions for critique
        provider_labels = list(range(1, len(providers) + 1))
        random.shuffle(provider_labels)
        
        for idx, provider in enumerate(providers):
            anonymized_answers[f"AI #{provider_labels[idx]}"] = initial_answers[provider]
        
        # Get critiques
        critiques = {}
        for i, provider in enumerate(providers):
            if provider not in llm_client.clients:
                critiques[provider] = f"API key not configured for {provider}"
                continue
                
            # Create critique prompt
            other_answers_list = []
            for label, answer in anonymized_answers.items():
                other_answers_list.append(f"{label}: {answer}")
            
            if other_answers_list:
                critique_prompt = f"""Original question: {prompt}
Anonymous AI responses (identifiers are randomly assigned):
{chr(10).join(other_answers_list)}
Please provide a constructive critique of the above anonymous responses. Focus on:
1. Accuracy and factual correctness
2. Completeness of the answer
3. Clarity and coherence
4. Any important points missed
5. Suggestions for improvement
Do NOT try to guess which AI generated which response. Just evaluate the responses objectively.
Your critique:"""
                
                critiques[provider] = llm_client.query_provider(provider, critique_prompt)
            else:
                critiques[provider] = "No other answers to critique"
            
            # Update critiques text
            critiques_text += f"## {provider}'s Critique\n{critiques.get(provider, 'No critique')}\n\n{'='*60}\n\n"
            yield initial_text, critiques_text, "⏳ Waiting for chairman..."
            progress(0.4 + (0.3 * (i + 1) / len(providers)), desc=f"Got critique from {provider}")
    else:
        critiques_text = "No critiques available (need at least 2 AIs for critiques)"
        yield initial_text, critiques_text, "⏳ Waiting for chairman..."
    
    # Step 3: Get chairman verdict
    progress(0.7, desc="Getting chairman verdict")
    chairman_verdict = "Chairman not configured"
    if chairman in llm_client.clients:
        # Prepare anonymized discussion summary
        discussion_summary = f"""Original Question: {prompt}
Anonymous AI Responses (identifiers are randomly assigned):
"""
        for label, answer in anonymized_answers.items():
            discussion_summary += f"\n{label}:\n{answer}\n"
            discussion_summary += "-" * 50 + "\n"
        
        if len(providers) > 1 and 'critiques' in locals():
            discussion_summary += "\n\nCritiques from Anonymous Reviewers:\n"
            critique_labels = list(range(1, len(critiques) + 1))
            random.shuffle(critique_labels)
            
            for idx, (provider, critique) in enumerate(critiques.items()):
                discussion_summary += f"\nCritique #{critique_labels[idx]}:\n{critique}\n"
                discussion_summary += "-" * 50 + "\n"
        
        chairman_prompt = f"""{discussion_summary}
As the Chairman AI, your task is to synthesize the above anonymous discussion and provide a final verdict:
1. Summarize the key points from all responses
2. Evaluate the strengths and weaknesses of each approach based on the critiques
3. Identify consensus points and disagreements
4. Provide the most accurate, comprehensive answer to the original question
5. Explain your reasoning for choosing this synthesis
IMPORTANT: You do not know which AI generated which response. The identifiers (AI #1, AI #2, etc.) are randomly assigned. Focus solely on the content quality.
Final Verdict:"""
        
        chairman_verdict = llm_client.query_provider(chairman, chairman_prompt)
    
    progress(1.0, desc="Discussion complete")
    yield initial_text, critiques_text, chairman_verdict

def update_api_keys(openai_key, anthropic_key, google_key, groq_key, perplexity_key,
                   openai_model, anthropic_model, google_model, groq_model, perplexity_model):
    """Update API keys for all providers - TEMPORARY IN-MEMORY ONLY"""
    updates = [
        ("OpenAI", openai_key, openai_model),
        ("Anthropic", anthropic_key, anthropic_model),
        ("Google", google_key, google_model),
        ("Groq", groq_key, groq_model),
        ("Perplexity", perplexity_key, perplexity_model)
    ]
    
    results = []
    for provider, key, model in updates:
        if key or model:
            success = llm_client.update_api_key(provider, key, model)
            status = "✅ Updated (in-memory only)" if success else "❌ Failed"
            results.append(f"{provider}: {status}")
    
    return "\n".join(results) if results else "No changes made"

def update_settings(temperature, max_tokens, active_providers, chairman_provider,
                    openai_model_setting, anthropic_model_setting, google_model_setting, 
                    groq_model_setting, perplexity_model_setting):
    """Update generation settings - TEMPORARY IN-MEMORY ONLY"""
    llm_client.update_settings(temperature, max_tokens)
    llm_client.update_active_providers(active_providers)
    llm_client.set_chairman_provider(chairman_provider)
    
    # Update model settings for each provider
    model_updates = [
        ("OpenAI", openai_model_setting),
        ("Anthropic", anthropic_model_setting),
        ("Google", google_model_setting),
        ("Groq", groq_model_setting),
        ("Perplexity", perplexity_model_setting)
    ]
    
    for provider, model in model_updates:
        if model:
            llm_client.update_api_key(provider, "", model)
    
    return "✅ Settings updated (in-memory only)!"

def load_current_settings():
    """Load current settings for the settings tab"""
    config = llm_client.config
    providers = config["providers"]
    
    return (
        "",  # Empty string for API keys - never show stored values
        "",  # Empty string for API keys - never show stored values
        "",  # Empty string for API keys - never show stored values
        "",  # Empty string for API keys - never show stored values
        "",  # Empty string for Perplexity API key
        providers["OpenAI"]["model"],
        providers["Anthropic"]["model"],
        providers["Google"]["model"],
        providers["Groq"]["model"],
        providers["Perplexity"]["model"],
        config["temperature"],
        config["max_tokens"],
        config["active_providers"],
        config["chairman_provider"],
        providers["OpenAI"]["model"],  # For settings tab dropdown
        providers["Anthropic"]["model"],  # For settings tab dropdown
        providers["Google"]["model"],  # For settings tab dropdown
        providers["Groq"]["model"],  # For settings tab dropdown
        providers["Perplexity"]["model"]  # For settings tab dropdown
    )

# Create Gradio interface
with gr.Blocks(title="LLMs-Assemble App") as app:
    gr.Markdown("# LLMs-Assemble App")
    gr.Markdown("Query multiple LLM providers simultaneously with your prompt")
    
    with gr.Tab("💬 Chat"):
        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Enter your prompt",
                    placeholder="What is the capital of France?",
                    lines=5
                )
                submit_btn = gr.Button("Submit", variant="primary")
                
                with gr.Accordion("⚙️ Generation Settings", open=False):
                    temperature = gr.Slider(
                        minimum=0, maximum=2, value=0.7, step=0.1,
                        label="Temperature"
                    )
                    max_tokens = gr.Slider(
                        minimum=100, maximum=4000, value=1000, step=100,
                        label="Max Tokens"
                    )
            
            with gr.Column(scale=3):
                # Show all models by default in chat (all tabs visible)
                outputs = []
                for provider in SUPPORTED_PROVIDERS:
                    with gr.Tab(provider):
                        output = gr.Textbox(
                            label=f"{provider} Response",
                            lines=15,
                            interactive=False,
                            max_lines=50,
                            autoscroll=True
                        )
                        outputs.append(output)
        
        # Use the new function with progressive updates
        submit_btn.click(
            fn=process_query_with_updates,
            inputs=[prompt_input, temperature, max_tokens],
            outputs=outputs
        )
        
        # Example prompts
        gr.Examples(
            examples=[
                "Explain quantum computing in simple terms",
                "Write a short poem about artificial intelligence",
                "What are the benefits of renewable energy?",
                "Create a recipe for vegan chocolate cake"
            ],
            inputs=prompt_input
        )
    
    with gr.Tab("👨‍⚖️ Chairman Mode"):
        gr.Markdown("### Chairman AI Discussion")
        gr.Markdown("Select AI models to discuss a question, critique each other, and get a final verdict from the Chairman")
        
        with gr.Row():
            with gr.Column(scale=1):
                chairman_prompt = gr.Textbox(
                    label="Discussion Question",
                    placeholder="Should we prioritize AI safety research over AI capabilities research?",
                    lines=5
                )
                
                selected_providers = gr.CheckboxGroup(
                    label="Select AI Participants",
                    choices=SUPPORTED_PROVIDERS,
                    value=SUPPORTED_PROVIDERS
                )
                
                chairman_provider = gr.Dropdown(
                    label="Select Chairman AI",
                    choices=SUPPORTED_PROVIDERS,
                    value="OpenAI"
                )
                
                chairman_btn = gr.Button("Start Discussion", variant="primary")
                
                with gr.Accordion("⚙️ Discussion Settings", open=False):
                    chairman_temp = gr.Slider(
                        minimum=0, maximum=2, value=0.7, step=0.1,
                        label="Temperature"
                    )
                    chairman_tokens = gr.Slider(
                        minimum=100, maximum=4000, value=1500, step=100,
                        label="Max Tokens"
                    )
            
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Initial Answers"):
                        # Create a single textbox for initial answers
                        initial_output = gr.Textbox(
                            label="Initial Answers from Selected AIs",
                            lines=20,
                            interactive=False,
                            max_lines=100,
                            autoscroll=True
                        )
                    
                    with gr.Tab("Critiques"):
                        critiques_output = gr.Textbox(
                            label="AI Critiques (Anonymized)",
                            lines=20,
                            interactive=False,
                            max_lines=100,
                            autoscroll=True
                        )
                    
                    with gr.Tab("Chairman Verdict"):
                        chairman_output = gr.Textbox(
                            label="Final Verdict (Based on Anonymous Evaluation)",
                            lines=25,
                            interactive=False,
                            max_lines=100,
                            autoscroll=True
                        )
        
        # Use the new function with progressive updates
        chairman_btn.click(
            fn=process_chairman_discussion_with_updates,
            inputs=[chairman_prompt, selected_providers, chairman_provider, chairman_temp, chairman_tokens],
            outputs=[initial_output, critiques_output, chairman_output]
        )
        
        # Example prompts for chairman mode
        gr.Examples(
            examples=[
                "What are the ethical implications of advanced AI systems?",
                "Is universal basic income a good solution for AI-induced unemployment?",
                "How should we regulate AI development to balance innovation and safety?",
                "What's the most effective way to learn programming in 2024?"
            ],
            inputs=chairman_prompt,
            label="Example Discussion Topics"
        )
    
    with gr.Tab("🔑 API Key Management"):
        gr.Markdown("### Configure API Keys")
        gr.Markdown("**⚠️ Keys are stored in-memory only and will be lost when app restarts**")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### OpenAI")
                openai_key = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="sk-...",
                    value=""
                )
                openai_model = gr.Dropdown(
                    label="Model",
                    choices=OPENAI_MODELS,
                    value="gpt-4o-mini"
                )
                
                gr.Markdown("#### Anthropic")
                anthropic_key = gr.Textbox(
                    label="Anthropic API Key",
                    type="password",
                    placeholder="sk-ant-...",
                    value=""
                )
                anthropic_model = gr.Dropdown(
                    label="Model",
                    choices=ANTHROPIC_MODELS,
                    value="claude-opus-4.5"
                )
            
            with gr.Column():
                gr.Markdown("#### Google Gemini")
                google_key = gr.Textbox(
                    label="Google API Key",
                    type="password",
                    placeholder="AIza...",
                    value=""
                )
                google_model = gr.Dropdown(
                    label="Model",
                    choices=GOOGLE_MODELS,
                    value="gemini-2.5-flash"
                )
                
                gr.Markdown("#### Groq")
                groq_key = gr.Textbox(
                    label="Groq API Key",
                    type="password",
                    placeholder="gsk_...",
                    value=""
                )
                groq_model = gr.Dropdown(
                    label="Model",
                    choices=GROQ_MODELS,
                    value="grok-4"
                )
            
            with gr.Column():
                gr.Markdown("#### Perplexity")
                perplexity_key = gr.Textbox(
                    label="Perplexity API Key",
                    type="password",
                    placeholder="pplx-...",
                    value=""
                )
                perplexity_model = gr.Dropdown(
                    label="Model",
                    choices=PERPLEXITY_MODELS,
                    value="sonar-reasoning"
                )
        
        update_keys_btn = gr.Button("Update API Keys (In-memory only)", variant="primary")
        key_update_status = gr.Textbox(label="Status", interactive=False)
        
        update_keys_btn.click(
            fn=update_api_keys,
            inputs=[
                openai_key, anthropic_key, google_key, groq_key, perplexity_key,
                openai_model, anthropic_model, google_model, groq_model, perplexity_model
            ],
            outputs=key_update_status
        )
    
    with gr.Tab("⚙️ Settings"):
        gr.Markdown("### App Settings (In-memory only)")
        
        active_providers = gr.CheckboxGroup(
            label="Active Providers",
            choices=SUPPORTED_PROVIDERS,
            value=SUPPORTED_PROVIDERS
        )
        
        chairman_setting = gr.Dropdown(
            label="Default Chairman AI",
            choices=SUPPORTED_PROVIDERS,
            value="OpenAI"
        )
        
        temperature_setting = gr.Slider(
            minimum=0, maximum=2, value=0.7, step=0.1,
            label="Temperature"
        )
        
        max_tokens_setting = gr.Slider(
            minimum=100, maximum=4000, value=llm_client.config["max_tokens"], step=100,
            label="Max Tokens"
        )
        
        with gr.Accordion("🔄 Model Selection for Active Providers", open=True):
            gr.Markdown("Select models for each active provider")
            openai_model_setting = gr.Dropdown(
                label="OpenAI Model",
                choices=OPENAI_MODELS,
                value="gpt-4o-mini"
            )
            anthropic_model_setting = gr.Dropdown(
                label="Anthropic Model",
                choices=ANTHROPIC_MODELS,
                value="claude-opus-4.5"
            )
            google_model_setting = gr.Dropdown(
                label="Google Model",
                choices=GOOGLE_MODELS,
                value="gemini-2.5-flash"
            )
            groq_model_setting = gr.Dropdown(
                label="Groq Model",
                choices=GROQ_MODELS,
                value="grok-4"
            )
            perplexity_model_setting = gr.Dropdown(
                label="Perplexity Model",
                choices=PERPLEXITY_MODELS,
                value="sonar-reasoning"
            )
        
        update_settings_btn = gr.Button("Update Settings (In-memory only)", variant="primary")
        settings_status = gr.Textbox(label="Status", interactive=False)
        
        update_settings_btn.click(
            fn=update_settings,
            inputs=[
                temperature_setting, max_tokens_setting, active_providers, chairman_setting,
                openai_model_setting, anthropic_model_setting, google_model_setting,
                groq_model_setting, perplexity_model_setting
            ],
            outputs=settings_status
        )
        
        refresh_btn = gr.Button("Load Current Settings")
        refresh_btn.click(
            fn=load_current_settings,
            outputs=[
                openai_key, anthropic_key, google_key, groq_key, perplexity_key,
                openai_model, anthropic_model, google_model, groq_model, perplexity_model,
                temperature_setting, max_tokens_setting, active_providers, chairman_setting,
                openai_model_setting, anthropic_model_setting, google_model_setting,
                groq_model_setting, perplexity_model_setting
            ]
        )

if __name__ == "__main__":
    print(" Starting Multi-LLM Comparison App...")
    print("⚠️  API Keys Security Notice:")
    print("   - API keys loaded from environment variables only")
    print("   - No API keys are saved to disk")
    print("   - Keys entered in UI are stored in-memory only")
    print("🌐 Open http://localhost:7860 in your browser")
    
    app.launch(share=False)
