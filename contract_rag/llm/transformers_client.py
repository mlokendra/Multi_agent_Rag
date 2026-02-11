from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

#import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass(frozen=True)
class TransformersConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    max_new_tokens: int = 200
    temperature: float = 0.0
    top_p: float = 0.9
    seed: int = 42


class TransformersClient:
    """
    Local LLM client using HuggingFace Transformers.
    Default: Qwen/Qwen2.5-3B-Instruct.

    Compatible with Answerer via: complete(system, user) -> str
    """

    def __init__(self, config: Optional[TransformersConfig] = None) -> None:
        self.config = config or self._load_from_env()
        self._load_model()

    @staticmethod
    def _load_from_env() -> TransformersConfig:
        return TransformersConfig(
            model_name=os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
            max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "512")),
            temperature=float(os.getenv("HF_TEMPERATURE", "0.1")),
            top_p=float(os.getenv("HF_TOP_P", "0.9")),
            seed=int(os.getenv("HF_SEED", "42")),
        )

    def _load_model(self) -> None:
        
        torch.manual_seed(self.config.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )

        # Prefer 4-bit on CUDA; otherwise fall back to CPU full precision to avoid import errors.
        use_cuda = torch.cuda.is_available()
        quant_config = None
        device_map = "auto"
        dtype = None

        if use_cuda:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            device_map = {"": "cpu"}
            dtype = torch.float32

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quant_config,
                device_map=device_map,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        except Exception:
            # Retry on CPU without quantization (slow but reliable).
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map={"": "cpu"},
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )

        self.model.eval()


    def complete(self, system: str, user: str) -> str:
        """
        Uses Qwen chat template if available.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # Qwen models support chat templates in tokenizer
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Try to return only the assistant's last turn
        # (simple heuristic)
        if "assistant" in text.lower():
            return text.splitlines()[-1].strip()
        return text.strip()
