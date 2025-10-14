import os
import gc
import torch
import yaml
import json
from collections import OrderedDict
from safetensors.torch import save_file
from peft import LoraConfig


class AsDictLoader(yaml.SafeLoader):
    pass

def construct_any_python_object(loader, tag_suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


AsDictLoader.add_multi_constructor('tag:yaml.org,2002:python/object', construct_any_python_object)
AsDictLoader.add_multi_constructor('!python/object', construct_any_python_object)


def extract_lora_from_lightning_ckpt(ckpt_full_path: str, output_dir: str):
    """
    Extract LoRA weights directly from a full PyTorch Lightning checkpoint (.pt or .ckpt)
    and convert them into a Diffusers-compatible format.
    """
    if not os.path.exists(ckpt_full_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_full_path}")

    # === 1. Locate hparams.yaml ===
    print("🔍 Locating hparams.yaml ...")
    parts = ckpt_full_path.split(os.sep)
    try:
        idx = parts.index("checkpoints")
    except ValueError:
        raise ValueError(f"'checkpoints' not found in path: {ckpt_full_path}")
    root_path = os.sep.join(parts[:idx])
    hparams_path = os.path.join(root_path, "hparams.yaml")

    if not os.path.exists(hparams_path):
        raise FileNotFoundError(f"hparams.yaml not found: {hparams_path}")

    # === 2. Load hyperparameters ===
    with open(hparams_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=AsDictLoader)

    args = data.get("args", {})
    rank = args.get("rank", 4)
    lora_cfg = data.get("lora_config", {})

    print(f"📘 LoRA rank: {rank}")

    # === 3. Load full checkpoint ===
    print(f"📦 Loading checkpoint: {ckpt_full_path}")
    checkpoint = torch.load(ckpt_full_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["module"]
    del checkpoint
    gc.collect()

    # === 4. Extract transformer-related parameters ===
    print("🧩 Extracting transformer weights ...")
    transformer_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("flux_transformer."):
            new_key = k.replace("flux_transformer.", "")
            transformer_state_dict[new_key] = v
        elif k.startswith("transformer."):
            new_key = k.replace("transformer.", "")
            transformer_state_dict[new_key] = v

    if len(transformer_state_dict) == 0:
        raise ValueError("No transformer weights found in checkpoint.")

    # === 5. Extract LoRA layers only ===
    print("🧮 Extracting LoRA parameters ...")
    lora_state_dict = {
        "transformer." + k.replace("default.", ""): v
        for k, v in transformer_state_dict.items()
        if "lora_" in k or "lora_A" in k or "lora_B" in k
    }

    if len(lora_state_dict) == 0:
        raise ValueError("No LoRA parameters detected. Please check your checkpoint structure.")

    print(f"✅ Found {len(lora_state_dict)} LoRA parameters")

    # === 6. Build LoRA config ===
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_cfg.get("lora_alpha", 4),
        target_modules=lora_cfg.get("target_modules", ["attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0"]),
        init_lora_weights=lora_cfg.get("init_lora_weights", True),
        bias=lora_cfg.get("bias", "none"),
        lora_dropout=lora_cfg.get("lora_dropout", 0.0),
    )

    # === 7. Save as safetensors + config ===
    os.makedirs(output_dir, exist_ok=True)
    lora_path = os.path.join(output_dir, "adapter_model.safetensors")
    config_path = os.path.join(output_dir, "adapter_config.json")

    save_file(lora_state_dict, lora_path)
    print(f"💾 Saved LoRA weights to: {lora_path}")

    def make_json_safe(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_safe(v) for v in obj]
        else:
            return obj

    config_dict = make_json_safe(lora_config.to_dict())
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"💾 Saved LoRA config to: {config_path}")
    print("\n🎉 Done! LoRA extraction complete.\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python get_lora_weights.py <path_to_your_pt_file> [output_dir]")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "lora_output"

    extract_lora_from_lightning_ckpt(
        ckpt_full_path=ckpt_path,
        output_dir=output_dir
    )
