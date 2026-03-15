"""LLM-as-judge evaluation for generated text quality.

Uses an external LLM API (OpenAI-compatible endpoint) to score generated
texts on fluency, coherence, and instruction-following.
"""

from __future__ import annotations

import json
import logging

import requests

logger = logging.getLogger(__name__)


class LLMJudge:
    """Evaluate generated texts using an external LLM as a judge.

    Sends each text to an OpenAI-compatible chat completions endpoint and
    parses numeric scores for fluency, coherence, and instruction-following.
    """

    DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"
    DEFAULT_MODEL = "gpt-4o-mini"

    SYSTEM_PROMPT = (
        "You are an expert text quality evaluator. For each text provided, "
        "rate it on three dimensions using a 1-5 scale:\n"
        "- fluency: grammatical correctness and natural flow\n"
        "- coherence: logical consistency and topical focus\n"
        "- instruction_following: how well the text follows any implicit or "
        "explicit instructions\n\n"
        "Respond ONLY with a JSON object: "
        '{{"fluency": <int>, "coherence": <int>, "instruction_following": <int>}}'
    )

    def __init__(
        self,
        api_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the LLM judge.

        Args:
            api_url: OpenAI-compatible chat completions URL.
            model: Model identifier to use for evaluation.
        """
        self.api_url = api_url or self.DEFAULT_API_URL
        self.model = model or self.DEFAULT_MODEL

    def evaluate(
        self,
        generated_texts: list[str],
        api_key: str,
    ) -> list[dict]:
        """Evaluate a list of generated texts via the external LLM API.

        Args:
            generated_texts: Texts to evaluate.
            api_key: API key for authentication.

        Returns:
            List of dicts, each with keys ``fluency``, ``coherence``,
            ``instruction_following`` (int scores 1-5). On API or parse
            failure the dict contains an ``error`` key instead.
        """
        results: list[dict] = []

        for text in generated_texts:
            result = self._evaluate_single(text, api_key)
            results.append(result)

        return results

    def _evaluate_single(self, text: str, api_key: str) -> dict:
        """Send a single text to the LLM API and parse the score response."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Evaluate this text:\n\n{text}"},
            ],
            "temperature": 0.0,
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            scores = json.loads(content)

            # Validate expected keys
            for key in ("fluency", "coherence", "instruction_following"):
                if key not in scores:
                    raise KeyError(f"Missing key: {key}")
                scores[key] = int(scores[key])

            return {
                "fluency": scores["fluency"],
                "coherence": scores["coherence"],
                "instruction_following": scores["instruction_following"],
            }

        except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("LLM judge evaluation failed: %s", exc)
            return {"error": str(exc)}
