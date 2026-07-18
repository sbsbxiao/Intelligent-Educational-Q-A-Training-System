from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from config import settings

logger = logging.getLogger("agent_hub.local_text_generation")


@dataclass
class LocalTextGenerationConfig:
    model_path: str
    max_new_tokens: int = 512


class LocalTextGenerationModel:
    def __init__(self, config: LocalTextGenerationConfig) -> None:
        self.config = config
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: str = "cpu"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self._ensure_loaded()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    async def agenerate(self, system_prompt: str, user_prompt: str) -> str:
        return await asyncio.to_thread(self.generate, system_prompt, user_prompt)

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Local text generation model path does not exist: {self.config.model_path}")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if self._device == "cuda" else torch.float32
        logger.info("Loading local text generation model: %s on %s", self.config.model_path, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self._device)
        self._model.eval()


def create_local_text_generation_model() -> LocalTextGenerationModel:
    return LocalTextGenerationModel(
        LocalTextGenerationConfig(
            model_path=settings.local_text_generation_path,
            max_new_tokens=settings.local_text_generation_max_new_tokens,
        )
    )
