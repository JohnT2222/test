from __future__ import annotations

import os
import argparse
from pathlib import Path
from llm_client import LLMClient


import json
import argparse
from typing import Any, Dict, Optional


def _trim_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1 :]
    return s.strip()


def _coerce_json(resp: Any) -> Dict[str, Any]:
    if isinstance(resp, dict):
        return resp
    if isinstance(resp, str):
        return json.loads(_trim_code_fences(resp))
    return json.loads(str(resp))


def _spec_schema(n_models: int, n_vars: int) -> Dict[str, Any]:
    """
    JSON schema the LLM must follow.
    """
    return {
        "type": "object",
        "properties": {
            "suggested_models": {
                "type": "array",
                "minItems": max(1, min(n_models, 8)),
                "maxItems": max(1, min(n_models, 8)),
                "items": {
                    "type": "object",
                    "required": ["name", "rationale", "task_type"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "rationale": {"type": "string", "minLength": 1},
                        "task_type": {
                            "type": "string",
                            "enum": [
                                "regression",
                                "classification",
                                "ranking",
                                "clustering",
                                "time-series",
                                "dimensionality-reduction",
                                "causal-inference",
                                "reinforcement-learning",
                                "other",
                            ],
                        },
                        "when_to_use": {"type": "string"},
                        "complexity": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                    "additionalProperties": False,
                },
            },
            "independent_variables": {
                "type": "array",
                "minItems": max(3, min(n_vars, 15)),
                "maxItems": max(3, min(n_vars, 15)),
                "items": {
                    "type": "object",
                    "required": ["name", "rationale"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "rationale": {"type": "string", "minLength": 1},
                        "expected_direction": {
                            "type": "string",
                            "enum": ["positive", "negative", "nonlinear", "unknown"],
                        },
                        "measurement_idea": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "required": ["suggested_models", "independent_variables"],
        "additionalProperties": False,
    }


def _instruction(query: str, n_models: int, n_vars: int, schema: Dict[str, Any]) -> str:
    return (
        "You are a modeling strategist. From the user's query, propose:\n"
        f"- Exactly {n_models} candidate MODEL approaches (diverse where sensible).\n"
        f"- Exactly {n_vars} INDEPENDENT VARIABLES that could serve as predictors/features.\n\n"
        "Rules:\n"
        "- Return ONLY strict JSON matching the provided schema (no prose, no comments, no markdown).\n"
        "- Keep names short and concrete (e.g., 'VO2 max', 'EEG Alpha Power', 'Vertical Jump').\n"
        "- 'rationale' should be 1–2 sentences max.\n"
        "- If uncertain about directionality, set 'expected_direction' to 'unknown'.\n"
        "- For models, set 'task_type' appropriately (e.g., regression, classification, time-series, etc.) and include 'when_to_use' if helpful.\n"
        "- Avoid dataset-specific claims; focus on generalizable, defensible suggestions.\n\n"
        "Return JSON only with keys: 'suggested_models' and 'independent_variables'.\n\n"
        f"Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"User query:\n{query}"
    )


class TextSpecRetriever:
    """
    Speculates modeling approaches and independent variables from a natural-language query.
    Uses LLMClient (provider='stub' or 'groq') and returns STRICT structured JSON.
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> None:
        self.llm = llm or LLMClient(provider=provider, model=model, temperature=temperature, max_tokens=max_tokens)

    def run(self, query: str, n_models: int = 3, n_vars: int = 8) -> Dict[str, Any]:
        n_models = max(1, min(int(n_models), 8))
        n_vars = max(3, min(int(n_vars), 15))
        schema = _spec_schema(n_models, n_vars)

        # Build a single-turn prompt with schema included; the LLMClient handles the call.
        system_msg = (
            "You are a precise modeling strategist. Output must be strictly valid JSON following the given schema."
        )
        user_msg = _instruction(query=query, n_models=n_models, n_vars=n_vars, schema=schema)

        # We’ll reuse LLMClient.chat for a single JSON response, or better, use a JSON function.
        # Since LLMClient already has a JSON-returning helper (speculate_features), we’ll emulate that logic here.
        # Compose a chat call with system+user to reduce drift.
        resp = self.llm._client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.llm.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
        ) if getattr(self.llm, "_client", None) else _stub_response(n_models, n_vars)

        content = resp.choices[0].message.content if isinstance(resp, object) and hasattr(resp, "choices") else resp
        data = _coerce_json(content)

        # Minimal validation (shape)
        if "suggested_models" not in data or "independent_variables" not in data:
            raise ValueError("Response missing required keys: 'suggested_models' and/or 'independent_variables'.")

        return data


def _stub_response(n_models: int, n_vars: int) -> Dict[str, Any]:
    # Deterministic-ish stub for offline dev
    models = [
        {
            "name": "Linear Regression",
            "rationale": "Simple baseline for continuous outcomes; coefficients are interpretable.",
            "task_type": "regression",
            "when_to_use": "Start here when relationships look roughly linear.",
            "complexity": "low",
        },
        {
            "name": "Random Forest",
            "rationale": "Nonlinear model robust to outliers and mixed feature types.",
            "task_type": "classification",
            "when_to_use": "When you expect interactions and nonlinearity without heavy tuning.",
            "complexity": "medium",
        },
        {
            "name": "Gradient Boosting (XGBoost)",
            "rationale": "Strong tabular performance with feature importance insights.",
            "task_type": "regression",
            "when_to_use": "When you need high accuracy on structured data.",
            "complexity": "medium",
        },
        {
            "name": "Time-Series Forecasting (Prophet)",
            "rationale": "Captures trend and seasonality with minimal configuration.",
            "task_type": "time-series",
            "when_to_use": "For timestamped outcomes with trend/seasonality.",
            "complexity": "low",
        },
    ][:n_models]

    base_vars = [
        {"name": "VO2 max", "rationale": "Captures aerobic capacity linked to endurance.", "expected_direction": "positive", "measurement_idea": "Treadmill test or wearable estimation."},
        {"name": "Grip Strength", "rationale": "Proxy for general strength and neuromuscular health.", "expected_direction": "positive", "measurement_idea": "Hand dynamometer."},
        {"name": "Reaction Time (ms)", "rationale": "Reflects neural processing speed relevant to agility.", "expected_direction": "negative", "measurement_idea": "On-screen tap test latency."},
        {"name": "HRV (RMSSD)", "rationale": "Indicator of recovery and autonomic balance.", "expected_direction": "positive", "measurement_idea": "Wearable HRV during rest."},
        {"name": "EEG Alpha Power", "rationale": "Correlates with relaxed focus and cognitive readiness.", "expected_direction": "nonlinear", "measurement_idea": "Consumer EEG headband protocols."},
        {"name": "Vertical Jump", "rationale": "Explosive lower-body power.", "expected_direction": "positive", "measurement_idea": "Jump mat or camera-based estimation."},
        {"name": "Lactate Threshold", "rationale": "Predicts sustainable high-intensity output.", "expected_direction": "positive", "measurement_idea": "Progressive exertion protocol with blood sampling."},
        {"name": "1RM Deadlift", "rationale": "Max strength related to total body capacity.", "expected_direction": "positive", "measurement_idea": "Gym-based test with safety constraints."},
        {"name": "Stride Variability", "rationale": "Motor control metric related to agility/stability.", "expected_direction": "nonlinear", "measurement_idea": "Gait analysis from video or IMU."},
        {"name": "Sleep Efficiency", "rationale": "Recovery quality affecting performance.", "expected_direction": "positive", "measurement_idea": "Wearable or sleep diary-derived."},
    ][:n_vars]

    return {"suggested_models": models, "independent_variables": base_vars}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculate models and independent variables from a query.")
    parser.add_argument("--query", required=True, help="User query describing the prediction/analysis goal.")
    parser.add_argument("--n_models", type=int, default=3, help="How many model approaches to suggest (1..8).")
    parser.add_argument("--n_vars", type=int, default=8, help="How many independent variables to suggest (3..15).")
    parser.add_argument("--provider", type=str, default=None, help="Override LLM provider (stub/groq).")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model id.")
    args = parser.parse_args()

    retriever = TextSpecRetriever(provider=args.provider, model=args.model)
    out = retriever.run(query=args.query, n_models=args.n_models, n_vars=args.n_vars)
    print(json.dumps(out, indent=2))
