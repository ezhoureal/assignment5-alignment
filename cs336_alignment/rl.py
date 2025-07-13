import torch


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
    ratio = torch.exp(policy_log_probs - old_log_probs)
    advantages = advantages.expand_as(policy_log_probs)
    clipped_ratio = torch.where(
        advantages >= 0,
        torch.min(ratio, torch.Tensor([1.0 + cliprange])),
        torch.max(ratio, torch.Tensor([1.0 - cliprange])),
    )
    return (-clipped_ratio * advantages, {"clip_fraction": (clipped_ratio > 1.0).float().mean()})