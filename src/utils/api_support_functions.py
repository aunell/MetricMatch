import os
import requests
import time
from typing import Dict, Any
import tiktoken
import torch
import json
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

# Global variables for Qwen model (lazy loading)
_qwen_model = None
_qwen_tokenizer = None

def load_qwen_model(model_name="Qwen/Qwen2.5-7B-Instruct"):
    """Lazy load Qwen model and tokenizer."""
    global _qwen_model, _qwen_tokenizer
    if _qwen_model is None:
        print(f"Loading Qwen model: {model_name}")
        _qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Qwen model loaded successfully")
    return _qwen_model, _qwen_tokenizer

# Global variables for Gemma model (lazy loading)
_gemma_model = None
_gemma_tokenizer = None

def load_gemma_model(model_name="google/gemma-3-1b-it"):
    """Lazy load Gemma model and tokenizer."""
    global _gemma_model, _gemma_tokenizer
    if _gemma_model is None:
        print(f"Loading Gemma model: {model_name}")
        _gemma_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _gemma_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Gemma model loaded successfully")
    return _gemma_model, _gemma_tokenizer

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

def completion_with_backoff_qwen(**kwargs) -> Dict[str, Any]:
    """Completion function for Qwen models using local HuggingFace."""
    try:
        model_name = kwargs.get('model_name', 'Qwen/Qwen2.5-7B-Instruct')
        messages = kwargs['messages']
        max_tokens = kwargs.get('max_tokens', 1000)
        temperature = kwargs.get('temperature', 0.7)

        # Load model
        model, tokenizer = load_qwen_model(model_name)

        # Format messages for Qwen chat template
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
        print(f"Error in Qwen completion: {e}")
        return None

def completion_with_backoff_gemma(**kwargs) -> Dict[str, Any]:
    """Completion function for Gemma models using local HuggingFace."""
    try:
        model_name = kwargs.get('model_name', 'google/gemma-3-1b-it')
        messages = kwargs['messages']
        max_tokens = kwargs.get('max_tokens', 1000)
        temperature = kwargs.get('temperature', 0.7)

        # Load model and tokenizer
        model, tokenizer = load_gemma_model(model_name)

        # Apply chat template and tokenize
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
            )

        # Decode only the generated part (skip the input prompt)
        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

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
        print(f"Error in Gemma completion: {e}")
        return None

def completion_with_backoff_gemini(**kwargs) -> Dict[str, Any]:
    """Completion function for Gemini models using Stanford Healthcare API."""
    retry_count = 0
    while True:
        retry_count += 1
        try:
            url = "https://apim.stanfordhealthcare.org/gemini-25-pro/gemini-25-pro"
            headers = {
                "Ocp-Apim-Subscription-Key": OPENAI_API_KEY,
                "Content-Type": 'application/json'
            }
            # Extract text from messages - combine system and user messages
            messages = kwargs['messages']
            combined_text = ""
            for msg in messages:
                if msg['role'] == 'system':
                    combined_text += msg['content'] + "\n\n"
                elif msg['role'] == 'user':
                    combined_text += msg['content']

            data = {
                "contents": [
                    {
                    "role": "user",
                    "parts": [
                    {
                    "text": combined_text,
                    }
                    ]
                    }
                    ]
                }

            # Add generation config if temperature or max_tokens provided
            generation_config = {}
            if 'temperature' in kwargs and kwargs['temperature'] is not None:
                generation_config['temperature'] = kwargs['temperature']
            if 'max_tokens' in kwargs and kwargs['max_tokens'] is not None:
                generation_config['maxOutputTokens'] = kwargs['max_tokens']
            if generation_config:
                data['generationConfig'] = generation_config
            # Pass data directly to json parameter - don't double-encode with json.dumps
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

            # Transform Gemini response format to OpenAI format for compatibility
            gemini_response = response.json()

            # Handle case where response is a list (Gemini returns a list with one element)
            if isinstance(gemini_response, list) and len(gemini_response) > 0:
                gemini_response = gemini_response[0]

            # Debug: print the response structure if it's not what we expect
            try:
                text_content = gemini_response['candidates'][0]['content']['parts'][0]['text']
            except (KeyError, TypeError, IndexError) as e:
                print(f"Unexpected Gemini response structure: {e}")
                print(f"Response: {gemini_response}")
                # Try to handle error responses
                if isinstance(gemini_response, dict) and 'error' in gemini_response:
                    print(f"Gemini API error: {gemini_response.get('error')}")
                    return None
                raise

            return {
                'choices': [{
                    'message': {
                        'content': text_content
                    }
                }]
            }
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

DEEPSEEK_API_URL = "https://apim.stanfordhealthcare.org/deepseekr1/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-r1"

def completion_with_backoff_deepseek(**kwargs) -> Dict[str, Any]:
    """Completion function for DeepSeek R1 via Stanford Healthcare API."""
    retry_count = 0
    while True:
        retry_count += 1
        try:
            headers = {
                "Ocp-Apim-Subscription-Key": OPENAI_API_KEY,
                "Content-Type": 'application/json'
            }
            model_name = kwargs.get('model_name', DEEPSEEK_MODEL)
            data = {
                "model": model_name,
                "messages": kwargs['messages'],
                "max_tokens": kwargs.get('max_tokens', 1000),
                "temperature": kwargs.get('temperature', 0.7)
            }
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response text: {response.text}")
                if response.status_code == 400:
                    print("400 error encountered. Skipping this prompt.")
                    return None
                if response.status_code == 429:
                    print("429 Rate limit hit. Retrying after delay.")
                    time.sleep(10)
                    continue
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


def completion_with_backoff_llama33(**kwargs) -> Dict[str, Any]:
    """Completion function for Llama 3.3 70B Instruct via Stanford Healthcare API."""
    retry_count = 0
    while True:
        retry_count += 1
        try:
            url = f"https://aihubapi.stanfordhealthcare.org/azure-openai/deployments/llama-3-3-70b-instruct/chat/completions?api-version={API_VERSION}"
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
                if response.status_code == 400:
                    print("400 error encountered. Skipping this prompt.")
                    return None
                if response.status_code == 429:
                    print("429 Rate limit hit. Retrying after delay.")
                    time.sleep(10)
                    continue
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
            if model_name=="gpt-5":
                data = {
                "messages": kwargs['messages'],
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