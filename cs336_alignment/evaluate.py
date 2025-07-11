import json

def evaluate_vllm(
vllm_model: LLM,
reward_fn: Callable[[str, str], dict[str, float]],
prompts: List[str],
eval_sampling_params: SamplingParams
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    results = []
    for prompt in prompts:
        response = vllm_model.generate(prompt, **eval_sampling_params)
        reward = reward_fn(prompt, response)
        results.append((prompt, response, reward))
    # Serialize results to disk
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f)


eval_sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>",
                                                       include_stop_str_in_output=True]
)
llm = LLM(model="qwen/qwen-1.5b")
from datasets import load_dataset

ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")
prompts = [open("prompts/r1_zero.prompt").read() for _ in range(len(ds))]

evaluate_vllm(llm, drgrpo_grader.r1_zero_reward_fn, prompts, eval_sampling_params)