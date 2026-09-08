"""
Base framework abstraction providing:
- Pretrained loading (config + normalization stats + weights)
- Action space utilities (dimension, stats, (un)normalization)
- Trainable module discovery helper
Note: No device placement or optimizer concerns handled here (delegated to trainer).
"""

import importlib
import os
import pkgutil
from pathlib import Path
from typing import Any, Dict, List

# Force HuggingFace libraries to use local files only — prevents network timeouts
# on servers without internet access (each timeout costs 30-120s per from_pretrained call).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import random as _random
import torch

# Evaluation servers can opt into faster kernels.  The default remains the
# original deterministic behavior so training/reproducibility is unchanged.
_FAST_INFERENCE = os.environ.get("STARVLA_FAST_INFERENCE", "0") == "1"
if _FAST_INFERENCE:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # Let scaled-dot-product attention choose the fastest valid CUDA kernel.
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
else:
    # Deterministic mode — ensures reproducible results across runs.
    # flash_attention_2 + bfloat16 can produce non-bitwise-identical results
    # due to CuDNN kernel selection and floating-point non-associativity.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

_SEED = int(os.environ.get("STARVLA_SEED", "42"))
torch.manual_seed(_SEED)
torch.cuda.manual_seed_all(_SEED)
np.random.seed(_SEED)
_random.seed(_SEED)
from transformers import PretrainedConfig, PreTrainedModel

from starVLA.model.framework.share_tools import dict_to_namespace, read_mode_config
from starVLA.model.tools import FRAMEWORK_REGISTRY, FrameworkTools, auto_get_trainable_modules
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)
_FRAMEWORKS_IMPORTED = False


def _auto_import_framework_modules() -> None:
    global _FRAMEWORKS_IMPORTED
    if _FRAMEWORKS_IMPORTED:
        return

    _SKIP = {"__init__", "base_framework", "share_tools"}
    framework_dir = Path(__file__).resolve().parent

    # Scan top-level modules (backwards compat)
    for _, module_name, is_pkg in pkgutil.iter_modules([str(framework_dir)]):
        if module_name in _SKIP:
            continue
        if is_pkg:
            # Scan sub-packages (VLM4A/, WM4A/, etc.)
            sub_dir = framework_dir / module_name
            for _, sub_name, _ in pkgutil.iter_modules([str(sub_dir)]):
                if sub_name.startswith("_"):
                    continue
                importlib.import_module(f"starVLA.model.framework.{module_name}.{sub_name}")
        else:
            importlib.import_module(f"starVLA.model.framework.{module_name}")

    _FRAMEWORKS_IMPORTED = True


def build_framework(cfg): # The single entry point for building different model frameworks
    """
    Build a framework model from config.
    Args:
        cfg: Config object containing `cfg.framework.name`.
    Returns:
        nn.Module: Instantiated framework model.
    """
    if not hasattr(cfg, "framework") or not hasattr(cfg.framework, "name"):
        raise ValueError("Missing `cfg.framework.name`. The framework API now only accepts `framework.name`.")

    _auto_import_framework_modules()

    framework_id = cfg.framework.name
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        available = sorted(FRAMEWORK_REGISTRY._registry.keys())
        raise NotImplementedError(
            f"Framework `{framework_id}` is not implemented. Available frameworks: {available}"
        )

    model_class = FRAMEWORK_REGISTRY[framework_id]
    return model_class(cfg)


# PreTrainedModel, AutoModel, PretrainedConfig,  are so good, find sometime to study them
# TODO @JinhuiYE find sometime to merge yaml config with transformer config


class baseframework(PreTrainedModel):
    """
    Lightweight base class for higher-level VLA model assemblies.
    Subclasses are expected to:
      - Accept a structured config
      - Register components in __init__
      - Use provided helpers for action normalization handling
    """

    def __init__(self, hf_config=PretrainedConfig()) -> None:
        """
        Initialize base nn.Module. Subclasses add components.
        """

        super().__init__(hf_config)

    # ------------------------------------------------------------------
    # Soft-constraint interface: subclasses should override these.
    # Default implementations raise NotImplementedError so that IDE
    # tooling (e.g. pylance, mypy) flags missing overrides, while
    # still allowing PreTrainedModel instantiation (no ABC).
    # ------------------------------------------------------------------

    def forward(self, examples: List[dict], **kwargs) -> dict:
        """Training forward pass.

        Args:
            examples: List[dict], each dict requires at least:
                - image: List[PIL.Image]
                - lang: str
                - action: np.ndarray shaped [T, action_dim]

        Returns:
            dict: Must contain ``"action_loss"`` (torch.Tensor scalar).
                  May contain extra keys for logging (e.g. ``"kl_loss"``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward(examples) -> dict with 'action_loss' key."
        )

    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        """Inference: predict future actions from observations.

        Args:
            examples: Same schema as *forward* (minus ``action`` which is optional).
            **kwargs: Framework-specific inference options (e.g. ``use_ddim``).

        Returns:
            dict: Must contain ``"normalized_actions"`` (np.ndarray [B, T, action_dim]).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement predict_action(examples) -> dict with 'normalized_actions' key."
        )

    # ------------------------------------------------------------------
    # Unified loss interface for Trainer
    # ------------------------------------------------------------------

    def supports_training_tag(self, tag: str) -> bool:
        """Return whether this framework can consume batches for *tag*."""
        if tag == "vla":
            return type(self).forward is not baseframework.forward
        if tag == "vlm":
            return hasattr(self, "qwen_vl_interface") or type(self).forward_vlm is not baseframework.forward_vlm
        return False

    def compute_loss(self, tag: str, batch, loss_scale: dict = None) -> Dict[str, torch.Tensor] | None:
        """Unified forward entry-point: route to the right forward by *tag*.

        The trainer calls ``model.compute_loss(tag, batch)`` for every
        ``(tag, batch)`` pair produced by :class:`DataLoaderManager`.
        The model internally dispatches:

        - ``"vla"`` → ``self.forward(batch)``
        - ``"vlm"`` → ``self.forward_vlm(batch)``

        Subclasses can override this to add more tags (e.g. ``"world"``).

        Args:
            tag: dataset type tag (``"vla"``, ``"vlm"``, …)
            batch: the batch produced by the corresponding DataLoader.
            loss_scale: ``{"vla": 1.0, "vlm": 0.1}`` per-tag loss multiplier.
                        Defaults to 1.0 for unspecified tags.

        Returns:
            dict[str, Tensor] | None: keyed losses (e.g. ``{"action_loss": ...}``).
                Returns ``None`` when this framework does not support the
                incoming dataloader tag so the trainer can ``continue``.
        """
        if not self.supports_training_tag(tag):
            return None

        scale = (loss_scale or {}).get(tag, 1.0)

        if tag == "vla":
            out = self.forward(batch)
        elif tag == "vlm":
            out = self.forward_vlm(batch)
        else:
            return None

        # Apply loss scale and filter to Tensor values only
        return {k: v * scale for k, v in out.items() if isinstance(v, torch.Tensor)}

    def forward_vlm(self, batch) -> Dict[str, torch.Tensor]:
        """VLM forward pass (default implementation).

        Delegates to ``self.qwen_vl_interface(**batch)`` which is present on
        every framework subclass that uses a Qwen VL backbone.

        Subclasses may override to add custom VLM logic.

        Args:
            batch: dict produced by the VLM dataloader.

        Returns:
            dict: Must contain ``"vlm_loss"`` (torch.Tensor scalar).
        """
        if not hasattr(self, "qwen_vl_interface"):
            raise NotImplementedError(
                f"{type(self).__name__} has no `qwen_vl_interface`. "
                "Override forward_vlm() to support VLM training."
            )
        out = self.qwen_vl_interface(**batch)
        return {"vlm_loss": out.loss}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: str,
        **kwargs,
    ) -> None:
        """
        Restore a model instance from a saved checkpoint.

        Workflow:
            1. Resolve checkpoint path
            2. Load config + dataset normalization statistics
            3. Build model with loaded config
            4. Load state_dict strictly (reports missing/unexpected keys)
            5. Attach normalization stats for later un-normalization

        Args:
            pretrained_checkpoint: Path to .pt file inside run/checkpoints directory.
            **kwargs: Extra constructor overrides passed to subclass.

        Returns:
            baseframework: Instantiated model (left on CPU; caller decides device).

        Raises:
            RuntimeError: If state_dict key mismatch occurs under strict=True.
            FileNotFoundError: If underlying files are missing (surfaced earlier).
        """
        pretrained_checkpoint = Path(pretrained_checkpoint)
        model_config, norm_stats = read_mode_config(pretrained_checkpoint)  # read config and norm_stats

        config = dict_to_namespace(model_config)
        model_config = config
        model_config.trainer.pretrained_checkpoint = None
        
        FrameworkModel = build_framework(cfg=model_config)
        # set for action un-norm
        FrameworkModel.norm_stats = norm_stats
        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        if pretrained_checkpoint.suffix == ".safetensors":
            from safetensors.torch import load_file

            model_state_dict = load_file(str(pretrained_checkpoint))
        else:
            # mmap=True: memory-map the 39-43GB checkpoint instead of copying to RAM.
            # Tensors are read on demand, cutting load time and peak memory significantly.
            model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu", mmap=True)
        # logger.info(f"Loading model weights from `{pretrained_checkpoint}`")
        # Cache state_dict ONCE — calling it inside the loop (2584 keys) re-traverses
        # the full model graph each time, adding minutes of overhead.
        model_sd = FrameworkModel.state_dict()
        model_keys = set(model_sd.keys())
        checkpoint_keys = set(model_state_dict.keys())
        # Filter out VLM keys that don't match (e.g., 2B vs 4B)
        compatible_state_dict = {}
        for k, v in model_state_dict.items():
            if k in model_keys:
                if v.shape == model_sd[k].shape:
                    compatible_state_dict[k] = v
                else:
                    logger.warning(f"Skipping key {k}: shape mismatch {v.shape} vs {model_sd[k].shape}")
            else:
                logger.debug(f"Skipping unexpected key: {k}")

        missing_keys = model_keys - set(compatible_state_dict.keys())
        if missing_keys:
            logger.warning(f"Missing keys (will be randomly initialized): {len(missing_keys)} keys")
            for mk in sorted(missing_keys)[:10]:
                logger.warning(f"  {mk}")

        FrameworkModel.load_state_dict(compatible_state_dict, strict=False)
        logger.info(f"Loaded {len(compatible_state_dict)}/{len(model_keys)} params")

        # **ensure model is on GPU**
        FrameworkModel = FrameworkModel
        return FrameworkModel
