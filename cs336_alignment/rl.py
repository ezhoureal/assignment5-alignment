from typing import Literal
import torch
from tqdm import tqdm

from cs336_alignment.evaluate import format_prompt


def compute_group_normalized_rewards(
reward_fn,
rollout_responses,
repeated_ground_truths,
group_size,
advantage_eps,
normalize_by_std,
):
    raw_reward = []
    for (i, (response, ground_truth)) in enumerate(zip(rollout_responses, repeated_ground_truths)):
        group = i // group_size
        if group == len(raw_reward):
            raw_reward.append([])
        raw_reward[group].append(reward_fn(response, ground_truth)["reward"])

    # divide into group size and compute mean/std
    raw_reward = torch.tensor(raw_reward, dtype=torch.float32)
    mean = torch.mean(raw_reward, dim=1, keepdim=True)
    if normalize_by_std:
        std = torch.std(raw_reward, dim=1, keepdim=True) + advantage_eps
        reward = (raw_reward - mean) / (std + advantage_eps)
    else:
        reward = raw_reward - mean
    return (torch.flatten(reward), torch.flatten(raw_reward), {})

def compute_naive_policy_gradient_loss(
raw_rewards_or_advantages: torch.Tensor,
policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the per-token GRPO-Clip loss.

    Args:
        advantages: Tensor of shape (batch_size, 1), per-example advantages.
        policy_log_probs: Tensor of shape (batch_size, sequence_length), per-token log probs from the new policy.
        old_log_probs: Tensor of shape (batch_size, sequence_length), per-token log probs from the old policy.
        cliprange: float, clip parameter ϵ (e.g., 0.2).

    Returns:
        Tuple containing:
        - loss: Tensor of shape (batch_size, sequence_length), the per-token clipped loss.
        - metadata: Dictionary containing logging information, such as whether each token was clipped.
    """
    # Compute the ratio of new policy probabilities to old policy probabilities
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # Broadcast advantages to match the sequence length dimension
    advantages_broadcasted = advantages.expand_as(policy_log_probs)

    # Compute the unclipped objective
    unclipped_objective = ratio * advantages_broadcasted

    # Compute the clipped objective
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    clipped_objective = clipped_ratio * advantages_broadcasted

    # Compute the GRPO-Clip loss
    loss = -torch.min(unclipped_objective, clipped_objective)

    # Determine which tokens were clipped
    clipped = (ratio < (1.0 - cliprange)) | (ratio > (1.0 + cliprange))

    # Prepare metadata for logging
    metadata = {
        "clipped": clipped.float(),  # Convert boolean to float for logging
        "ratio": ratio,
    }

    return loss, metadata

def compute_policy_gradient_loss(
policy_log_probs: torch.Tensor,
loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
raw_rewards: torch.Tensor | None= None,
advantages: torch.Tensor | None= None,
old_log_probs: torch.Tensor | None= None,
cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        return compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        ), {}
    elif loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        ), {
            "advantages": advantages,
        }
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Supported types are 'no_baseline', 'reinforce_with_baseline', and 'grpo_clip'.")
    
def masked_mean(
tensor: torch.Tensor,
mask: torch.Tensor,
dim: int | None= None,
) -> torch.Tensor:
    masked_tensor = tensor * mask
    return torch.sum(masked_tensor, dim=dim) / torch.sum(mask, dim=dim)

def grpo_microbatch_train_step(
policy_log_probs: torch.Tensor,
response_mask: torch.Tensor,
gradient_accumulation_steps: int,
loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
raw_rewards: torch.Tensor | None= None,
advantages: torch.Tensor | None= None,
old_log_probs: torch.Tensor | None= None,
cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, log = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    loss = masked_mean(loss, response_mask)
    if gradient_accumulation_steps > 1:
        loss /= gradient_accumulation_steps

    loss.backward()
    return loss, {**log, "response_mask": response_mask}

# script
n_grpo_steps: int = 200
learning_rate: float = 1e-5
advantage_eps: float = 1e-6
rollout_batch_size: int = 256
group_size: int = 4
sampling_temperature: float = 1.0
sampling_min_tokens: int = 4 # As in Expiter, disallow empty string responses
sampling_max_tokens: int = 1024
epochs_per_rollout_batch: int = 3 # On-policy
train_batch_size: int = 256 # On-policy
gradient_accumulation_steps: int = 128 # microbatch size is 2, will fit on H100
gpu_memory_utilization: float = 0.85
loss_type: Literal[
"no_baseline",
"reinforce_with_baseline",
"grpo_clip",
] = "reinforce_with_baseline"
use_std_normalization: bool = True

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from evaluate import *
from sft import *

model_name = "Qwen/Qwen3-0.6B"  # Use a smaller model for testing
tokenizer = AutoTokenizer.from_pretrained(model_name)
policy = AutoModelForCausalLM.from_pretrained(model_name)
optimizer = torch.optim.AdamW(
policy.parameters(),
lr=learning_rate,
weight_decay=0.0,
betas=(0.9, 0.95),
)
old_policy = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

reward_func = lambda response, ground_truth: {
    "reward": eval(response.replace('×', '*').replace('÷', '/')) == ground_truth
}

math_data = []
ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")
for item in ds["train"]:
    math_data.append(item)
for i in tqdm.trange(n_grpo_steps, desc="GRPO training steps"):
    if i % epochs_per_rollout_batch == 0:
        old_policy.load_state_dict(policy.state_dict())
    # randomly select data points
    math_data = ds["train"].shuffle(seed=42).select(range(rollout_batch_size))
    ground_truths = [item["target"] for item in math_data]
    prompts = [format_prompt(item) for item in math_data]
    # run ollama inference to obtain response
    response = [prompt + ollama_generate(model_name, prompt) for prompt in prompts]
    print(f'response = {response}')
    # feed response to policy to get log probabilities
    batch_sequence = tokenizer.encode(response)
    old_log_probs = old_policy(batch_sequence).logits
    old_log_probs = torch.nn.functional.log_softmax(old_log_probs, dim=-1)
    
    log_probs = policy(batch_sequence).logits
    log_probs = torch.nn.functional.log_softmax(log_probs, dim=-1) # repeat while output != </answer>

    advantage, raw_reward, log = compute_group_normalized_rewards(reward_fn=reward_func,
        rollout_responses=response, repeated_ground_truths=ground_truths,
        group_size=group_size, advantage_eps=advantage_eps,
        normalize_by_std=use_std_normalization)
    response_mask = [1 if (idx + 1) >= len(prompts) else 0 for idx in range(len(response) - 1)]
    loss, log = grpo_microbatch_train_step(
        policy_log_probs=log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        loss_type=loss_type,
        raw_rewards=raw_reward,
        advantages=advantage,
        old_log_probs=old_log_probs,
    )
    print(f'loss = {loss}')
    if i % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
       
