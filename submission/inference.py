"""
Inference script for the PrintFarmEnv v2 submission.

Supports two modes:
  1. Local model (Gemma 3 with optional LoRA adapter)
  2. OpenAI-compatible API (HF Inference Endpoints)

Environment variables:
  - MODEL_PATH: Path to local model or HF model name (default: google/gemma-3-1b-it)
  - ADAPTER_PATH: Path to LoRA adapter (default: None)
  - API_BASE_URL: For API mode (overrides local model)
  - MODEL_NAME: For API mode
  - HF_TOKEN / API_KEY: For API auth
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from submission.env.env import PrintFarmEnvironment
from submission.env.models import FarmAction, FarmActionEnum
from submission.shared.serialize import serialize_obs
from submission.shared.parse_action import parse_action, action_to_farm_action
from submission.shared.prompt import SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Global model state (loaded once)
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None
_backend = None


def _load_model():
    """Load model on first call."""
    global _model, _tokenizer, _backend

    if _model is not None:
        return

    model_path = os.getenv("MODEL_PATH", "google/gemma-3-1b-it")
    adapter_path = os.getenv("ADAPTER_PATH")

    # Check for adapter in submission directory
    if adapter_path is None:
        local_adapter = Path(__file__).resolve().parent / "adapter"
        if local_adapter.exists():
            adapter_path = str(local_adapter)

    try:
        from unsloth import FastLanguageModel
        _model, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path or model_path,
            max_seq_length=3500,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(_model)
        _backend = "unsloth"
        print(f"[inference] Loaded with Unsloth: {model_path}")
        return
    except ImportError:
        pass

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    _model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device,
    )

    if adapter_path:
        from peft import PeftModel
        _model = PeftModel.from_pretrained(_model, adapter_path)
        _model = _model.merge_and_unload()

    _model.eval()
    _backend = "transformers"
    print(f"[inference] Loaded: {model_path} on {device}")


def _use_api():
    """Check if we should use API mode."""
    return bool(os.getenv("API_BASE_URL"))


def _api_extract(state_json: str) -> FarmAction:
    """Use OpenAI-compatible API."""
    from openai import OpenAI

    api_base = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME", "google/gemma-3-1b-it")
    api_key = (os.getenv("HF_TOKEN")
               or os.getenv("OPENAI_API_KEY")
               or os.getenv("API_KEY")
               or "dummy_key")

    client = OpenAI(base_url=api_base, api_key=api_key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Current State:\n{state_json}"},
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=256,
            temperature=0.3,
        )
        text = response.choices[0].message.content
    except Exception as e:
        print(f"[inference] API error: {e}")
        return FarmAction(action=FarmActionEnum.WAIT)

    parsed = parse_action(text)
    if parsed is None:
        return FarmAction(action=FarmActionEnum.WAIT)
    return action_to_farm_action(parsed)


def _local_extract(state_json: str) -> FarmAction:
    """Use local model."""
    _load_model()

    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Current State:\n{state_json}"},
    ]

    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    inputs = _tokenizer(prompt, return_tensors="pt")
    if hasattr(_model, 'device'):
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    completion = _tokenizer.decode(new_tokens, skip_special_tokens=True)

    parsed = parse_action(completion)
    if parsed is None:
        return FarmAction(action=FarmActionEnum.WAIT)
    return action_to_farm_action(parsed)


def extract_action(state_json: str) -> FarmAction:
    """Main entry point — extract action from observation JSON string."""
    if _use_api():
        return _api_extract(state_json)
    return _local_extract(state_json)
