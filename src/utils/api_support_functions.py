import os
import requests
import time
from typing import Dict, Any
import tiktoken
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Azure OpenAI settings
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
API_VERSION = "2023-05-15"
print(f"OPENAI_API_BASE: {OPENAI_API_BASE}")
print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")

def format_evaluation(evaluation):
    """Format the evaluation data in a robust manner and extract scores."""
    score = evaluation['evaluation']['score']
    
    if score is None:
        # Attempt to find score in the raw evaluation
        for key, value in evaluation.items():
            if 'score' in key.lower():
                score = extract_score(value)
                break
    
    return {"score": score}

def get_tokenizer():
    return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))

def completion_with_backoff_anthropic(**kwargs) -> Dict[str, Any]:
    retry_count = 0
    while True:
        retry_count += 1
        try:
            url = "https://apim.stanfordhealthcare.org/Claude35Sonnetv2/awssig4fa"
            headers = {
                "Ocp-Apim-Subscription-Key": OPENAI_API_KEY,
                "Content-Type": 'application/json'
            }
            data = kwargs["messages"]
            # Add temperature if provided
            if "temperature" in kwargs and kwargs["temperature"] is not None:
                if isinstance(data, dict):
                    data["temperature"] = kwargs["temperature"]
            response = requests.post(url, headers=headers, json=data)
            try:
                response.raise_for_status()
            except:
                print(f"Error: {response.text}")
                if retry_count > 3:
                    return {}
                time.sleep(10)
                continue
            return response.json()
        except requests.exceptions.RequestException as error:
            print(f"Error: {error}")
            if hasattr(error, 'response') and error.response is not None:
                print("Response text:", error.response.text)
                if getattr(error.response, 'status_code', None) == 400:
                    print("400 error encountered. Skipping this prompt.")
                    return None
            if retry_count > 3:
                return {}
            time.sleep(10)

# Global variables for Llama model (lazy loading)
_llama_model = None
_llama_tokenizer = None

def load_llama_model(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    """Lazy load Llama model and tokenizer."""
    global _llama_model, _llama_tokenizer
    if _llama_model is None:
        print(f"Loading Llama model: {model_name}")
        _llama_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _llama_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Llama model loaded successfully")
    return _llama_model, _llama_tokenizer

def completion_with_backoff_llama(**kwargs) -> Dict[str, Any]:
    """Completion function for Llama models using local HuggingFace."""
    try:
        model_name = kwargs.get('model_name', 'meta-llama/Llama-3.1-8B-Instruct')
        messages = kwargs['messages']
        max_tokens = kwargs.get('max_tokens', 1000)
        temperature = kwargs.get('temperature', 0.7)

        # Load model
        model, tokenizer = load_llama_model(model_name)

        # Format messages for Llama chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode
        response_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Try to fix incomplete JSON by adding missing closing braces
        response_text = response_text.strip()
        if response_text.startswith('{'):
            # Count opening and closing braces
            open_braces = response_text.count('{')
            close_braces = response_text.count('}')
            # Add missing closing braces
            if open_braces > close_braces:
                # Add newline for better formatting before closing braces
                if not response_text.endswith('\n'):
                    response_text += '\n'
                response_text += '}' * (open_braces - close_braces)

        # Format response to match OpenAI structure
        return {
            'choices': [{
                'message': {
                    'content': response_text
                }
            }]
        }
    except Exception as e:
        print(f"Error in Llama completion: {e}")
        return None

def completion_with_backoff(**kwargs) -> Dict[str, Any]:
    retry_count = 0
    model_name = kwargs.get('model_name', 'gpt-4o')  # Default to gpt-4o if not specified
    while True:
        retry_count += 1
        try:
            url = f"{OPENAI_API_BASE}/deployments/{model_name}/chat/completions?api-version={API_VERSION}"
            headers = {
                "Ocp-Apim-Subscription-Key": OPENAI_API_KEY,
                "Content-Type": 'application/json'
            }
            data = {
                "messages": kwargs['messages'],
                "max_tokens": kwargs.get('max_tokens', 1000),
                "temperature": kwargs.get('temperature', 0.7)
            }
            response = requests.post(url, headers=headers, json=data)
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response text: {response.text}")
                # If it's a 400 error, likely a content policy violation or malformed request, skip this prompt
                if response.status_code == 400:
                    print("400 error encountered. Skipping this prompt.")
                    return None
                # If it's a 429 error, rate limit, retry
                if response.status_code == 429:
                    print("429 Rate limit hit. Retrying after delay.")
                    time.sleep(10)
                    continue
                # For other errors, retry up to 3 times
                if retry_count > 3:
                    return {}
                time.sleep(10)
                continue
            return response.json()
        except requests.exceptions.RequestException as error:
            print(f"Error: {error}")
            if hasattr(error, 'response') and error.response is not None:
                print("Response text:", error.response.text)
                if getattr(error.response, 'status_code', None) == 400:
                    print("400 error encountered. Skipping this prompt.")
                    return None
            if retry_count > 3:
                return {}
            time.sleep(10)


def extract_score(value):
    """Extract a numeric score from a value, handling various formats."""
    if isinstance(value, (int, float)):
        return value
    elif isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return None
    return None