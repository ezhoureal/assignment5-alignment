import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List

def tokenize_prompt_and_output(prompt_strs: List[str], output_strs: List[str], tokenizer) -> Dict[str, torch.Tensor]:
    batch_input_ids = []
    batch_sequence = []
    batch_response_mask = []
    
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)
        
        entire_seq = prompt_ids + output_ids
        response_mask_example = [1 if (idx + 1) >= len(prompt_ids) else 0 for idx in range(len(entire_seq) - 1)]
        
        batch_sequence.append(torch.tensor(entire_seq, dtype=torch.long))
        batch_response_mask.append(torch.tensor(response_mask_example, dtype=torch.long))

    batch_sequence = pad_sequence(batch_sequence, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_response_mask = pad_sequence(batch_response_mask, batch_first=True, padding_value=0)
    
    return {
        "input_ids": batch_sequence[:, :-1],
        "labels": batch_sequence[:, 1:],
        "response_mask": padded_response_mask
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # Compute log probabilities using log_softmax for numerical stability
    log_probs = torch.functional.F.log_softmax(logits, dim=-1)
    
    # Compute probabilities
    probs = torch.exp(log_probs)
    
    # Compute entropy: -sum(p * log(p)) over the vocabulary dimension
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    return entropy

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    # select log_prob of the label index
    output = {
        "log_probs": log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1),
    }
    if return_token_entropy:
        output["token_entropy"] = compute_entropy(logits)
    return output

def masked_normalize(
tensor: torch.Tensor,
mask: torch.Tensor,
normalize_constant: float,
dim: int | None= None,
) -> torch.Tensor:
    if dim is None:
        dim = tuple(range(tensor.dim()))
    masked_tensor = tensor * mask
    masked_sum = masked_tensor.sum(dim=dim)
    # Normalize
    return masked_sum / normalize_constant

def sft_microbatch_train_step(
policy_log_probs: torch.Tensor,
response_mask: torch.Tensor,
gradient_accumulation_steps: int,
normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=-1)
    loss = torch.mean(loss)
    if gradient_accumulation_steps > 1:
        loss /= gradient_accumulation_steps
    # Compute the gradients
    loss.backward()
    return loss, {"sum": sum}