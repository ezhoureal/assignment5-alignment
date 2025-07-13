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