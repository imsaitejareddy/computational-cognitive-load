import os
import sys
import json
import torch
import pandas as pd
import time
import google.generativeai as genai
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
import gc
import argparse

warnings.filterwarnings("ignore")

# --- Global variable for local model ---
local_model = None
tokenizer = None

def load_local_model(model_id, token):
    """Function to load a local model and clear GPU memory."""
    global local_model, tokenizer
    if local_model is not None:
        del local_model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Unloaded previous model to free GPU memory.")
    
    print(f"Loading {model_id}. This may take some time...")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    local_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        quantization_config=quantization_config, token=token
    )
    print(f"Successfully loaded {model_id}.")

def build_prompt(condition, qa_item, filler, distractor, irrelevance_percent=50):
    docs, prompt = list(qa_item["source_docs"].values()), "You are an expert financial analyst...\\n\\n"
    if condition == 'Control':
        for doc in docs: prompt += f"--- DOCUMENT ---\\n{doc}\\n\\n"
    elif condition == 'LongContextControl':
        for doc in docs: prompt += f"--- DOCUMENT ---\\n{doc}\\n\\n--- ANALYST COMMENTARY ---\\n{(filler['relevant_masked'] * 10)[:2000]}\\n\\n"
    elif condition == 'ContextSaturation':
        for doc in docs: prompt += f"--- DOCUMENT ---\\n{doc}\\n\\n--- ECONOMIC NEWS ---\\n{(filler['irrelevant'] * 10)[:int(4000 * (irrelevance_percent / 100))]}\\n\\n"
    elif condition == 'AttentionalResidue':
        for doc in docs: prompt += f"--- DOCUMENT ---\\n{doc}\\n\\n--- INTERMEDIATE TASK ---\\n{distractor['task']}\\n\\n--- END TASK ---\\n\\n"
    prompt += f"FINAL QUESTION: {qa_item['question']}"
    return prompt

def run_trial(model_name, prompt, openai_client):
    global local_model, tokenizer
    if "gpt-4" in model_name or "gemini" in model_name:
        if "gpt-4" in model_name:
            try:
                response = openai_client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], max_tokens=150, temperature=0.0)
                return response.choices[0].message.content
            except Exception as e: return f"API_ERROR: {e}"
        else:
            try:
                safety_settings = {"HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE", "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE", "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"}
                gemini_model = genai.GenerativeModel(model_name)
                response = gemini_model.generate_content(prompt, safety_settings=safety_settings)
                return response.text
            except Exception as e: return f"API_ERROR: {e}"
    else:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        local_model.eval()
        with torch.no_grad():
            outputs = local_model.generate(input_ids, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

def main(args):
    # --- Environment Setup ---
    print("Setting up API keys from environment variables...")
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    HF_TOKEN = os.environ.get('HF_TOKEN')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    OPENAI_PROJECT_ID = os.environ.get('OPENAI_PROJECT_ID')

    if not all([GEMINI_API_KEY, HF_TOKEN, OPENAI_API_KEY, OPENAI_PROJECT_ID]):
        raise ValueError("One or more API keys or Project ID not found. Please set them as environment variables.")

    genai.configure(api_key=GEMINI_API_KEY)
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT_ID)

    # --- Load Data ---
    with open('data/task_data.json', 'r') as f:
        task_data = json.load(f)
    qa_item = task_data['qa_item']
    filler = task_data['filler_text']
    distractor = task_data['distractor_task']

    # --- Main Experiment Loop ---
    results = []
    models_to_test = ["gpt-4o", "gemini-1.5-pro", "meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Meta-Llama-3-70B-Instruct"]
    conditions = ['Control', 'LongContextControl', 'ContextSaturation', 'AttentionalResidue']
    
    print(f"Starting definitive experiment with {args.num_replications} replications...")
    
    for model_name in models_to_test:
        print(f"\n{'='*20}\nTesting Model: {model_name}\n{'='*20}")
        if "gpt-4" not in model_name and "gemini" not in model_name:
            load_local_model(model_name, HF_TOKEN)

        for i in range(args.num_replications):
            print(f"--- Replication {i+1}/{args.num_replications} ---")
            for condition in conditions:
                if "gpt-4" in model_name or "gemini" in model_name: time.sleep(1)
                
                if condition == 'ContextSaturation':
                    for percent in [20, 50, 80]:
                        prompt = build_prompt(condition, qa_item, filler, distractor, irrelevance_percent=percent)
                        output = run_trial(model_name, prompt, openai_client)
                        results.append({"model": model_name, "replication": i, "condition": condition, "irrelevant_percent": percent, "output": output, "ground_truth": qa_item['ground_truth']})
                else:
                    prompt = build_prompt(condition, qa_item, filler, distractor)
                    output = run_trial(model_name, prompt, openai_client)
                    results.append({"model": model_name, "replication": i, "condition": condition, "irrelevant_percent": 0, "output": output, "ground_truth": qa_item['ground_truth']})
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_file, index=False)
    print(f"\nExperiment complete. Results saved to {args.output_file}")
    print(results_df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Interleaved Cognitive Evaluation (ICE) benchmark.")
    parser.add_argument('--num_replications', type=int, default=10, help='Number of replications to run for each condition.')
    parser.add_argument('--output_file', type=str, default='results/experiment_results.csv', help='Path to save the output CSV file.')
    args = parser.parse_args()
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    main(args)
