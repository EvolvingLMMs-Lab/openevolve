"""
Model ensemble for LLMs
"""

import asyncio
import logging
import random
from typing import Dict, List, Optional, Tuple

from openevolve.llm.base import LLMInterface
from openevolve.llm.gemini import GeminiLLM
from openevolve.llm.openai import OpenAILLM
from openevolve.config import LLMModelConfig
from pymongo import AsyncMongoClient
import os
import datetime



logger = logging.getLogger(__name__)


class LLMEnsemble:
    """Ensemble of LLMs"""

    def __init__(self, models_cfg: List[LLMModelConfig]):
        self.models_cfg = models_cfg

        # Initialize models from the configuration
        self.models = []
        for model_cfg in models_cfg:
            # Determine model type based on provider or model name
            if hasattr(model_cfg, 'provider') and model_cfg.provider == 'gemini':
                self.models.append(GeminiLLM(model_cfg))
            else: # default fallback to openai
                # Default to OpenAI for backward compatibility
                self.models.append(OpenAILLM(model_cfg))
        
        mongodb_uri = os.getenv('MONGODB_URI')
        logger.info(f"MONGODB_URI: {mongodb_uri}")
        if not mongodb_uri:
            raise ValueError("MONGODB_URI is not set")
        self.client = AsyncMongoClient(mongodb_uri)

        # Extract and normalize model weights
        self.weights = [model.weight for model in models_cfg]
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        logger.info(
            f"Initialized LLM ensemble with models: "
            + ", ".join(
                f"{model.name} (weight: {weight:.2f})"
                for model, weight in zip(models_cfg, self.weights)
            )
        )

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using a randomly selected model based on weights"""
        model = self._sample_model()
        result = await model.generate(prompt, **kwargs)
        await self.client.llm_responses.responses.insert_one({
            "model": model.model,
            "prompt": prompt,
            "result": result,
            "created_at": datetime.datetime.now(tz=datetime.timezone.utc),
            "configs": kwargs,
            "source": "openevolve"
        })
        return result

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        model = self._sample_model()
        result = await model.generate_with_context(system_message, messages, **kwargs)
        try:
            await self.client.llm_responses.responses.insert_one({
                "model": model.model,
                "system_message": system_message,
                "messages": messages,
                "result": result,
                "created_at": datetime.datetime.now(tz=datetime.timezone.utc),
                "configs": kwargs,
                "source": "openevolve"
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
        return result

    def _sample_model(self) -> LLMInterface:
        """Sample a model from the ensemble based on weights"""
        index = random.choices(range(len(self.models)), weights=self.weights, k=1)[0]
        return self.models[index]

    async def generate_multiple(self, prompt: str, n: int, **kwargs) -> List[str]:
        """Generate multiple texts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for _ in range(n)]
        return await asyncio.gather(*tasks)

    async def parallel_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def generate_all_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a all available models and average their returned metrics"""
        responses = []
        for model in self.models:
            responses.append(await model.generate_with_context(system_message, messages, **kwargs))
        return responses
