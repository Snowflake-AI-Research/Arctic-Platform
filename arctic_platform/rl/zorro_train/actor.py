"""
Actor implementation with prompt deduplication.

Provides forward and backward passes with automatic prompt deduplication.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from transformers import AutoModelForCausalLM

from .zorro_train import ZoRRoTrain
from .qwen_model_patcher import Qwen3ModelPatcher


class DeduplicatedActor:
    """Actor with prompt deduplication for forward pass."""
    
    def __init__(self, model_name_or_path: str, device: str = "cuda", patcher_class=None, use_split_attention: bool = True, attn_implementation: str = "sdpa"):
        """
        Initialize actor with prompt deduplication.
        
        Args:
            model_name_or_path: Hugging Face model identifier or local path
            device: Device to load model on
            patcher_class: Custom attention patcher class (default: auto-detect from model name)
            use_split_attention: Whether to use split attention optimization
            attn_implementation: Attention implementation (sdpa, flash_attention_2, flash_attention_3, eager)
        """
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        ).to(device)
        self.model.eval()
        
        # Auto-detect patcher class if not provided
        if patcher_class is None:
            patcher_class = self._auto_detect_patcher(model_name_or_path)
        
        self.patcher_class = patcher_class
        self.use_split_attention = use_split_attention
    
    def _auto_detect_patcher(self, model_name_or_path):
        """Auto-detect appropriate patcher class based on model name."""
        model_name_lower = model_name_or_path.lower()
        
        # Check for Qwen models
        if 'qwen' in model_name_lower:
            from .qwen_model_patcher import Qwen3ModelPatcher
            print(f"Auto-detected Qwen model, using Qwen3ModelPatcher")
            return Qwen3ModelPatcher
        
        # Default to base patcher
        print(f"Using default ModuleReconstructionPatcher")
        assert False, "Default patcher not implemented"
        
    def train(self):
        """Set model to training mode."""
        self.model.train()
        
    def eval(self):
        """Set model to eval mode."""
        self.model.eval()
        
    def _forward_micro_batch(
        self,
        micro_batch: Dict[str, torch.Tensor],
        temperature: float = 1.0,
        calculate_entropy: bool = False
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass with prompt deduplication.
        
        Args:
            micro_batch: Dict containing:
                - input_ids: [batch_size, seq_len]
                - position_ids: [batch_size, seq_len]
                - responses: [batch_size, response_len]
            temperature: Temperature for scaling logits
            calculate_entropy: Whether to compute entropy
            
        Returns:
            entropy: [batch_size, response_len] or None
            log_probs: [batch_size, response_len]
        """
        input_ids = micro_batch["input_ids"].to(self.device)
        position_ids = micro_batch["position_ids"].to(self.device)
        responses = micro_batch["responses"].to(self.device)
        
        response_length = responses.size(-1)
        batch_size, seq_len = input_ids.shape
        
        # Step 1: Find prompt groups
        prompt_groups, unique_prompts = ZoRRoTrain.find_prompt_groups(
            input_ids, response_length
        )
        
        print(f"Original batch size: {batch_size}")
        print(f"Number of unique prompts: {len(prompt_groups)}")
        for i, group in enumerate(prompt_groups):
            print(f"  Group {i}: {len(group)} samples sharing prompt")
        
        # Step 2: Create deduplicated batch
        dedup_input_ids, _, reconstruction_info = \
            ZoRRoTrain.create_deduplicated_batch(
                input_ids, position_ids,
                response_length, prompt_groups, unique_prompts
            )
        
        print(f"Deduplicated batch size: {dedup_input_ids.size(0)}")
        
        # Step 3: Forward pass with monkey-patched attention
        # Note: position_ids should NOT be deduplicated - use original position_ids
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            # Pass use_split_attention if patcher supports it
            patcher_kwargs = {'model': self.model, 'reconstruction_info': reconstruction_info}
            if hasattr(self.patcher_class, '__init__') and 'use_split_attention' in self.patcher_class.__init__.__code__.co_varnames:
                patcher_kwargs['use_split_attention'] = self.use_split_attention
            
            with self.patcher_class(**patcher_kwargs):
                output = self.model(
                    input_ids=dedup_input_ids,
                    position_ids=position_ids,  # Use original position_ids, not deduplicated
                    use_cache=False,
                )
        
        # Step 4: Get logits (already reconstructed by the patcher!)
        logits = output.logits  # [batch_size, seq_len, vocab_size]
        
        print(f"Model output logits shape: {logits.shape}")
        
        # Store full logits and reconstruction_info for testing/comparison and backward pass
        self._last_reconstructed_logits = logits.detach().clone()
        self._last_reconstruction_info = reconstruction_info
        
        prompt_len = reconstruction_info['prompt_len']
        
        # Step 5: Extract logits that predict response tokens
        # Response tokens are at [prompt_len : prompt_len + response_length]
        # Logits that predict them are at [prompt_len - 1 : prompt_len + response_length - 1]
        logits = logits[:, prompt_len - 1: prompt_len + response_length - 1, :]  # [batch_size, response_length, vocab_size]
        logits = logits / temperature
        
        # Compute log probabilities
        log_probs_all = torch.log_softmax(logits, dim=-1)
        log_probs = torch.gather(
            log_probs_all,
            dim=-1,
            index=responses.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, response_len]
        
        entropy = None
        if calculate_entropy:
            # Compute entropy: -sum(p * log(p))
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * log_probs_all).sum(dim=-1)  # [batch_size, response_len]
        
        return entropy, log_probs
    
    def compute_policy_loss_and_backward(
        self,
        micro_batch: Dict[str, torch.Tensor],
        temperature: float = 1.0,
        gradient_accumulation: int = 1,
    ) -> Dict[str, float]:
        """
        Compute PPO policy loss with deduplication and perform backward pass.
        Based on the original verl PPO implementation.
        
        Args:
            micro_batch: Dict containing:
                - input_ids: [batch_size, seq_len]
                - position_ids: [batch_size, seq_len]
                - responses: [batch_size, response_len]
                - response_mask: [batch_size, response_len] - mask for valid response tokens
                - old_log_probs: [batch_size, response_len] - log probs from rollout
                - advantages: [batch_size, response_len] - computed advantages
                - ref_log_prob: [batch_size, response_len] - reference policy log probs (optional)
            temperature: Temperature for scaling logits
            gradient_accumulation: Number of gradient accumulation steps
            
        Returns:
            metrics: Dict with loss values and statistics
        """
        # Get log probs from forward pass
        _, log_prob = self._forward_micro_batch(
            micro_batch, 
            temperature=temperature,
            calculate_entropy=False
        )
        
        # Extract fields from micro_batch
        old_log_prob = micro_batch["old_log_probs"].to(self.device)
        advantages = micro_batch["advantages"].to(self.device)
        
        # Response mask: if not provided, assume all tokens are valid (no padding)
        if "response_mask" in micro_batch:
            response_mask = micro_batch["response_mask"].to(self.device)
        else:
            # No padding - all response tokens are valid
            response_mask = torch.ones_like(log_prob)
        
        # Compute policy loss (PPO clipped objective)
        # Based on: verl/trainer/ppo/core_algos.py::compute_policy_loss
        
        # Log probability ratio: pi_theta / pi_theta_old
        log_ratio = log_prob - old_log_prob
        ratio = torch.exp(log_ratio)
        
        # Clipped objective
        clip_ratio = 0.2  # Default PPO clip ratio
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        pg_loss_element = torch.maximum(pg_loss1, pg_loss2)
        
        # Aggregate loss (token-mean by default)
        # Based on: verl/trainer/ppo/core_algos.py::agg_loss
        valid_tokens = response_mask.sum()
        pg_loss = (pg_loss_element * response_mask).sum() / (valid_tokens + 1e-8)
        
        policy_loss = pg_loss
        
        # Add KL loss if ref_log_prob is provided
        kl_loss = None
        if "ref_log_prob" in micro_batch:
            ref_log_prob = micro_batch["ref_log_prob"].to(self.device)
            
            # KL divergence (using low_var_kl / k3 approximation by default)
            # Based on: verl/trainer/ppo/core_algos.py::kl_penalty
            kld = log_prob - ref_log_prob
            kl_loss = (kld * response_mask).sum() / (valid_tokens + 1e-8)
            
            # Add to policy loss
            kl_loss_coef = 0.001  # Default coefficient
            policy_loss = policy_loss + kl_loss * kl_loss_coef
        
        # Scale for gradient accumulation
        loss = policy_loss / gradient_accumulation
        
        # Backward pass with patching active (for gradient checkpointing)
        patcher_kwargs = {'model': self.model, 'reconstruction_info': self._last_reconstruction_info}
        if hasattr(self.patcher_class, '__init__') and 'use_split_attention' in self.patcher_class.__init__.__code__.co_varnames:
            patcher_kwargs['use_split_attention'] = self.use_split_attention
        
        with self.patcher_class(**patcher_kwargs):
            loss.backward()
        
        # Compute metrics
        metrics = {
            "actor/pg_loss": pg_loss.detach().item(),
            "actor/policy_loss": policy_loss.detach().item(),
            "actor/loss": loss.detach().item(),
        }
        
        # Compute clipfrac (fraction of ratios that were clipped)
        clipfrac = ((ratio - 1.0).abs() > clip_ratio).float()
        clipfrac = (clipfrac * response_mask).sum() / (valid_tokens + 1e-8)
        metrics["actor/pg_clipfrac"] = clipfrac.detach().item()
        
        # Compute approximate KL (for monitoring, different from KL loss)
        approx_kl = ((ratio - 1.0) - log_ratio) * response_mask
        approx_kl = approx_kl.sum() / (valid_tokens + 1e-8)
        metrics["actor/ppo_kl"] = approx_kl.detach().item()
        
        if kl_loss is not None:
            metrics["actor/kl_loss"] = kl_loss.detach().item()
        
        return metrics

