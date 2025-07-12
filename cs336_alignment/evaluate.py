import json
import re
from typing import Callable, Dict, List, Any
import requests  # For Ollama's REST API

# need to run `ollama serve` first
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Default Ollama API endpoint

def make_r1_zero_prompt(question: str) -> str:
    """Format problem using r1_zero-shot prompt template"""
    return f"A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: {question}\nAssistant: <think>"

def ollama_generate(
    model: str,
    prompt: str,
    temperature: float = 1.0,
) -> str:
    """Generate text using Ollama's API"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # Disable streaming for simplicity
        "options": {
            "temperature": temperature,
            # "num_predict": max_tokens,
            "stop": ["</answer>"],  # Stop generation at this token
        },
    }
    
    response = requests.post(OLLAMA_API_URL, json=payload)
    response.raise_for_status()  # Raise error if API call fails
    return response.json()["response"]

import re

def extract_answer(completion: str, include_closing_tag: bool = False) -> str:
    """
    Extract everything after <answer> in the completion.
    
    Args:
        completion: Raw LLM output string
        include_closing_tag: Whether to keep </answer> in the result
    
    Returns:
        str: All text after <answer> (optionally including the closing tag)
    """
    pattern = r'<answer>(.*)'
    match = re.search(pattern, completion, re.DOTALL)
    
    if not match:
        return ""
    result = match.group(1).strip()
    if not include_closing_tag:
        # Remove closing tag if present
        result = result.replace('</answer>', '')
    return result

def evaluate_ollama(
    model: str,
    prompts: List[str],
    references: List[int],
    output_path: str = "math_eval_results.jsonl",
) -> Dict[str, float]:
    """
    Evaluate a model via Ollama on a list of prompts, compute metrics, and save results.
    """
    results = []
    metrics_agg = {
        "correct": 0,
        "incorrect": 0,
        "total": len(prompts),
    }
    for prompt, ref in zip(prompts, references):
        fullResponse = ollama_generate(model, prompt)
        answer = extract_answer(fullResponse)
        try:
            correct: bool = eval(answer.replace('×', '*').replace('÷', '/')) == ref
        except Exception as e:
            print(f"Eval error for completion: {fullResponse}\nError: {e}")
            correct = False
        results.append({
            # "prompt": prompt,
            "answer": answer,
            "reference": ref,
        })
        if correct:
            metrics_agg["correct"] += 1
        else:
            metrics_agg["incorrect"] += 1

    # Calculate aggregate metrics
    n = len(prompts)
    final_metrics = {k: v / n for k, v in metrics_agg.items()}

    # Calculate aggregate metrics
    n = len(prompts)
    final_metrics = {k: v / n for k, v in metrics_agg.items()}
    
    # Serialize results
    with open(output_path, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    
    return final_metrics

from datasets import load_dataset

def format_prompt(example):
    return (
        f"Use the numbers {', '.join(map(str, example['nums']))} "
        f"and the operations +, -, ×, ÷ to reach the target {example['target']}. "
        "Please only output the operation so that the result can be automatically evaluated."
    )

def main():
    # 1. Load MATH validation data
    math_data = []
    ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")
    for item in ds["train"]:
        math_data.append(item)
    math_data = math_data[8:10]  # Limit to first 10 examples for testing
    # 2. Prepare prompts and references
    prompts = [make_r1_zero_prompt(format_prompt(item)) for item in math_data]

    # 3. Run evaluation via Ollama
    metrics = evaluate_ollama(
        model="qwen3:0.6b",
        prompts=prompts,
        references=[item["target"] for item in math_data],
        output_path="qwen3_0.6b_math_eval_ollama.jsonl",
    )
    
    # 4. Print results
    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()