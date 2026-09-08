# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args

deepspeed_plugin = DeepSpeedPlugin(hf_ds_config="starVLA/config/deepseeds/ds_config.yaml")
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """Prepare co-training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    vlm_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vlm_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return vla_train_dataloader, vlm_train_dataloader


class VLAMTrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, vlm_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vlm_train_dataloader = vlm_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

        # Phase 3.1 diagnostics
        self._diag_enabled = (
            getattr(cfg.trainer, 'diagnostics_enabled', False)
            if cfg and cfg.trainer else False
        )
        self._diag_attn_ratio_vla = 0.0
        self._diag_attn_ratio_wm = 0.0
        self._diag_feat_norm_vla = 0.0
        self._diag_feat_norm_wm = 0.0
        self._diag_grad_cross_attn = 0.0
        self._diag_grad_other = 0.0
        self._diag_action_jitter = 0.0

        # DeepSpeed ZeRO manages gradients in internal buffers (p.grad is None
        # on the PyTorch module). We capture gradient norms via backward hooks
        # that fire during backward() before DeepSpeed takes over.
        self._grad_hooks_registered = False
        self._grad_norms_ca = []
        self._grad_norms_other = []

    def _register_grad_hooks(self):
        """Register backward hooks on trainable parameters to capture grad norms."""
        if self._grad_hooks_registered:
            return
        raw_model = self.accelerator.unwrap_model(self.model)

        def _make_hook(name):
            def _hook(grad):
                if grad is not None:
                    gn = grad.detach().norm().item()
                    if 'attn1' in name and any(x in name for x in ['to_q', 'to_k', 'to_v']):
                        self._grad_norms_ca.append(gn)
                    elif 'attn2' not in name:  # skip self-attn params
                        self._grad_norms_other.append(gn)
                return grad
            return _hook

        for name, p in raw_model.named_parameters():
            if p.requires_grad:
                p.register_hook(_make_hook(name))

        self._grad_hooks_registered = True
        logger.info(f"[GRAD] Registered backward hooks for gradient diagnostics")

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so a later setup-step crash still
        # leaves a from_pretrained-able run dir behind.
        self._save_initial_configs()

        if hasattr(self.config.trainer, "pretrained_checkpoint") and self.config.trainer.pretrained_checkpoint:
            pretrained_checkpoint = self.config.trainer.pretrained_checkpoint
            reload_modules = (
                self.config.trainer.reload_modules if hasattr(self.config.trainer, "reload_modules") else None
            )
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)

            # If checkpoint path contains step info (e.g. steps_4000_pytorch_model.pt)
            # AND is_resume is True, extract step number and resume from that point.
            # For new training phases loading pretrained weights, start from step 0.
            import re
            is_resume = getattr(self.config.trainer, "is_resume", False)
            _resume_match = re.search(r'steps_(\d+)', str(pretrained_checkpoint))
            if _resume_match and is_resume:
                self.completed_steps = int(_resume_match.group(1))
                # Restore LR scheduler position so cosine schedule continues correctly
                self.lr_scheduler.last_epoch = self.completed_steps
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"🔄 Resuming from step {self.completed_steps}")
            elif _resume_match:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"🆕 New training run loading pretrained weights from step {_resume_match.group(1)}, starting from step 0")

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        # Phase 1 WM warm-up: selective unfreeze for projector + cross-attn
        wm_warmup = (
            self.config.trainer.wm_warmup
            if (self.config and hasattr(self.config.trainer, "wm_warmup"))
            else False
        )
        if wm_warmup:
            self.model = self.setup_wm_warmup(self.model)

        # Phase 2 Action Head Fine-tuning: full unfreeze of action head + CFG mask token
        action_head_finetune = (
            self.config.trainer.action_head_finetune
            if (self.config and hasattr(self.config.trainer, "action_head_finetune"))
            else False
        )
        if action_head_finetune:
            self.model = self.setup_action_head_finetune(self.model)

        # Phase 3 Cross-Attention Fine-tuning: attention-mask CFG + WM dropout + EMA
        cross_attn_finetune = (
            self.config.trainer.cross_attn_finetune
            if (self.config and hasattr(self.config.trainer, "cross_attn_finetune"))
            else False
        )
        if cross_attn_finetune:
            self.model = self.setup_cross_attn_finetune(self.model)

        # EMA initialization (Phase 3)
        self.ema_enabled = (
            getattr(self.config.trainer, 'ema_enabled', False)
            if self.config and self.config.trainer else False
        )
        self.ema_params = None
        if self.ema_enabled:
            ema_decay = float(getattr(self.config.trainer, 'ema_decay', 0.999))
            self.ema_params = self.init_ema(self.model, decay=ema_decay)

        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader, self.vlm_train_dataloader = (
            self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
                self.vlm_train_dataloader,
            )
        )

        self._init_wandb()
        self._init_checkpointing()

        # Register gradient hooks AFTER DeepSpeed has wrapped the model
        if self._diag_enabled:
            self._register_grad_hooks()

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"\U0001f4dd Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"\U0001f4ca Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """Initialize checkpoint directory."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)

        if pretrained_checkpoint and is_resume and hasattr(self.config.trainer, "resume_from_checkpoint"):
            self._load_checkpoint(self.config.trainer.resume_from_checkpoint)

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            # Save EMA checkpoint
            if self.ema_enabled and self.ema_params is not None:
                ema_path = checkpoint_path + "_ema_pytorch_model.pt"
                self.save_ema_checkpoint(
                    self.accelerator.unwrap_model(self.model),
                    self.ema_params, ema_path
                )
                self.accelerator.print(f"✅ EMA checkpoint saved at {ema_path}")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and dist.get_rank() == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)
        self.vlm_iter = iter(self.vlm_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        vlm_loss_scale = getattr(self.config.trainer.loss_scale, "vlm", 1.0)
        if vlm_loss_scale == 0.0:
            batch_vlm = None
        else:
            try:
                batch_vlm = next(self.vlm_iter)
            except StopIteration:
                if not hasattr(self, "vlm_epoch_count"):
                    self.vlm_epoch_count = 0
                self.vlm_iter, self.vlm_epoch_count = self._reset_dataloader(self.vlm_train_dataloader, self.vlm_epoch_count)
                batch_vlm = next(self.vlm_iter)

        return batch_vla, batch_vlm

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla, batch_vlm = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla, batch_vlm)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model

            # Phase 3.1 diagnostics (only on logging steps)
            if self._diag_enabled and self.completed_steps % self.config.trainer.logging_frequency == 0:
                self._collect_phase3_diagnostics(batch_vla, step_metrics)

            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()
                dist.barrier()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Evaluate action prediction with current model."""
        if self.accelerator.is_main_process:
            examples, _ = self._get_next_batch()
            actions = [example["action"] for example in examples]

            output_dict = self.accelerator.unwrap_model(self.model).predict_action(examples=examples)
            normalized_actions = output_dict["normalized_actions"]

            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        dist.barrier()
        return step_metrics

    def _collect_phase3_diagnostics(self, batch_vla, log_dict):
        """Collect Phase 3.1 diagnostic metrics (only called on logging steps)."""
        if not self._diag_enabled or not self.accelerator.is_main_process:
            return

        try:
            unwrapped = self.accelerator.unwrap_model(self.model)
            examples = batch_vla

            # ── 1. Feature L2 Norm (VLA vs WM before cross-attn) ──
            imgs = [e["image"] for e in examples]
            instrs = [e["lang"] for e in examples]
            with torch.no_grad():
                vl_embs, _ = unwrapped._encode_vl_hidden_states(imgs, instrs)
                self._diag_feat_norm_vla = vl_embs[-1].norm(dim=-1).mean().item()

                wm_feat = None
                if unwrapped.world_model is not None and unwrapped.wm_projector is not None:
                    wi = unwrapped.world_model.build_inputs(imgs, instrs)
                    unwrapped.world_model(**wi)
                    wr = list(unwrapped.world_model._intermediate_features)
                    wm_feat = unwrapped.wm_projector(wr[-1])
                    self._diag_feat_norm_wm = wm_feat.norm(dim=-1).mean().item()

            # ── 2. Live Attention Ratio (hook ALL layers, pick cross-attn ones) ──
            # In this DiT, each block has only attn1 (no attn2).
            # Self-attn blocks: no encoder_hidden_states → kv_len == action_seq (~40)
            # Cross-attn blocks: encoder_hidden_states=VLA+WM → kv_len == VLA+WM (~309)
            # With interleave_self_attention=True, blocks alternate.
            # KEY: encoder_hidden_states is a KEYWORD arg, so we need with_kwargs=True.
            captured_attn = []

            def attn_hook(module, input, kwargs, output):
                encoder_hs = kwargs.get('encoder_hidden_states', None)
                q = module.to_q(input[0])
                k = module.to_k(encoder_hs if encoder_hs is not None else input[0])
                q = module.head_to_batch_dim(q)
                k = module.head_to_batch_dim(k)
                probs = module.get_attention_scores(q, k, attention_mask=None)
                captured_attn.append(probs.detach())

            blocks = unwrapped.action_model.model.transformer_blocks
            # Hook ALL blocks' attn1 with kwargs support
            handles = [b.attn1.register_forward_hook(attn_hook, with_kwargs=True) for b in blocks]

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    unwrapped.forward(examples)

            for h in handles:
                h.remove()

            if captured_attn and wm_feat is not None:
                vl_seq = vl_embs[0].shape[1]
                wm_seq = wm_feat.shape[1]
                ca_wm_ratios = []
                n_self_attn = 0
                for probs in captured_attn:
                    kv_len = probs.shape[-1]
                    # Cross-attention: kv_len ≈ vl_seq + wm_seq (~309)
                    # Self-attention:  kv_len ≈ action_seq (~40)
                    if kv_len < vl_seq + wm_seq * 0.5:  # self-attn, skip
                        n_self_attn += 1
                        continue
                    kv_attn = probs.sum(dim=0).sum(dim=0)
                    vla_sum = kv_attn[:vl_seq].sum().item()
                    wm_sum = kv_attn[min(vl_seq, kv_len):].sum().item()
                    total = vla_sum + wm_sum
                    if total > 0:
                        ca_wm_ratios.append(wm_sum / total)
                if ca_wm_ratios:
                    self._diag_attn_ratio_wm = sum(ca_wm_ratios) / len(ca_wm_ratios)
                    log_dict["diag/attn_ratio_wm_layers"] = len(ca_wm_ratios)
                if self.completed_steps <= 100:
                    logger.info(f"[DIAG] n_blocks={len(blocks)} n_self_attn={n_self_attn} "
                                f"n_cross_attn={len(ca_wm_ratios)} vl={vl_seq} wm={wm_seq} "
                                f"attn_ratio_wm={self._diag_attn_ratio_wm:.4f}")

            # ── 3. Action Smoothness (jitter) ──
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.float32):
                    pred = unwrapped.predict_action(examples=examples)
                    actions = pred["normalized_actions"]  # (B, H, D)
                    if actions.shape[1] > 1:
                        jitter = np.mean(np.abs(np.diff(actions, axis=1)))
                        self._diag_action_jitter = float(jitter)

        except Exception as e:
            import traceback
            logger.warning(f"Diagnostics collection failed: {e}\n{traceback.format_exc()}")

        # Log to metrics dict
        log_dict["diag/attn_ratio_wm"] = self._diag_attn_ratio_wm
        log_dict["diag/feat_norm_vla"] = self._diag_feat_norm_vla
        log_dict["diag/feat_norm_wm"] = self._diag_feat_norm_wm
        log_dict["diag/grad_norm_cross_attn"] = self._diag_grad_cross_attn
        log_dict["diag/grad_norm_other"] = self._diag_grad_other
        log_dict["diag/action_jitter"] = self._diag_action_jitter
        # Phase 3.2: value norms from forward pass
        if hasattr(unwrapped, '_phase3_diag') and unwrapped._phase3_diag:
            for k, v in unwrapped._phase3_diag.items():
                log_dict[f"diag/{k}"] = v

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm):
        """Execute single training step."""
        log_dict = {}
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            # Clear per-step accumulator for backward-hook-based gradient capture
            if self._diag_enabled:
                self._grad_norms_ca.clear()
                self._grad_norms_other.clear()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss
            self.accelerator.backward(total_loss)

            vlm_loss_scale = getattr(self.config.trainer.loss_scale, "vlm", 1.0)
            if batch_vlm is not None and vlm_loss_scale != 0.0:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    unwrapped = self.accelerator.unwrap_model(self.model)
                    vlm_output = unwrapped.qwen_vl_interface(**batch_vlm)
                    vlm_loss = vlm_output.loss * vlm_loss_scale
                self.accelerator.backward(vlm_loss)
                log_dict["vlm_loss"] = vlm_loss.item()
            else:
                log_dict["vlm_loss"] = 0.0

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            # ── Phase 3.1 diagnostics: gradient norms (hook-captured during backward) ──
            if self._diag_enabled and self.accelerator.sync_gradients:
                if self._grad_norms_ca:
                    self._diag_grad_cross_attn = sum(self._grad_norms_ca) / len(self._grad_norms_ca)
                if self._grad_norms_other:
                    self._diag_grad_other = sum(self._grad_norms_other) / len(self._grad_norms_other)
                if self.completed_steps <= 100 and self.completed_steps > 0:
                    logger.info(f"[GRAD] step={self.completed_steps} "
                                f"n_ca={len(self._grad_norms_ca)} n_other={len(self._grad_norms_other)} "
                                f"avg_ca={self._diag_grad_cross_attn:.6f} avg_other={self._diag_grad_other:.6f}")

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced.
            # See train_starvla.py for full explanation.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()
                # Phase 3: update EMA after each effective optimizer step
                if self.ema_enabled and self.ema_params is not None:
                    ema_decay = float(getattr(self.config.trainer, 'ema_decay', 0.999))
                    self.update_ema(
                        self.accelerator.unwrap_model(self.model),
                        self.ema_params,
                        ema_decay,
                    )

            log_dict.update(
                {
                    "action_dit_loss": action_loss.item(),
                }
            )

        return log_dict

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_interval = getattr(self.config.trainer, "save_interval", 0)
            if save_interval > 0 and self.completed_steps % save_interval == 0:
                logger.info(f"Training complete. Last step {self.completed_steps} already saved as checkpoint, skipping final_model.")
            else:
                save_format = getattr(self.config.trainer, "save_format", "pt")
                final_checkpoint = os.path.join(self.config.output_dir, "final_model")
                os.makedirs(final_checkpoint, exist_ok=True)
                state_dict = self.accelerator.get_state_dict(self.model)
                if save_format == "safetensors":
                    from safetensors.torch import save_file

                    save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
                elif save_format == "pt":
                    torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
                else:
                    raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
                logger.info(f"Training complete. Final model saved at {final_checkpoint}")

            # Save final EMA model
            if self.ema_enabled and self.ema_params is not None:
                ema_final_dir = os.path.join(self.config.output_dir, "final_ema_model")
                os.makedirs(ema_final_dir, exist_ok=True)
                self.save_ema_checkpoint(
                    self.accelerator.unwrap_model(self.model),
                    self.ema_params,
                    os.path.join(ema_final_dir, "pytorch_model.pt"),
                )
                logger.info(f"Training complete. Final EMA model saved at {ema_final_dir}")

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader, vlm_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLAMTrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        vlm_train_dataloader=vlm_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_收紧.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
