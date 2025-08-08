# file: [CONFIRM_PATH]\llm_client.py
from __future__ import annotations

import os, json, argparse
from typing import Any, Dict, Optional
from dotenv import load_dotenv

try:
    from groq import Groq  # pip install groq
except Exception:
    Groq = None  # type: ignore


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


def _default_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "features": {
                "type": "array",
                "minItems": 3,
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "required": ["name", "rationale", "weight"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "rationale": {"type": "string", "minLength": 1},
                        "weight": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "additionalProperties": False,
                },
            }
        },
        "required": ["features"],
        "additionalProperties": False,
    }


def _stub_speculate(n_features: int) -> Dict[str, Any]:
    base = [
        {"name": "VO2 max", "rationale": "Aerobic capacity relates to endurance.", "weight": 0.34},
        {"name": "Grip Strength", "rationale": "Proxy for general strength.", "weight": 0.22},
        {"name": "Reaction Time (ms)", "rationale": "Neural processing speed.", "weight": 0.16},
        {"name": "HRV (RMSSD)", "rationale": "Recovery and autonomic balance.", "weight": 0.14},
        {"name": "EEG Alpha Power", "rationale": "Relaxed focus readiness.", "weight": 0.14},
        {"name": "Vertical Jump", "rationale": "Explosive power marker.", "weight": 0.12},
    ]
    return {"features": (base * 2)[:max(3, min(n_features, 12))]}


class LLMClient:
    """
    Minimal, reusable LLM wrapper.
    - provider: 'stub' (offline) or 'groq'
    - model: Groq model id (default: qwen/qwen3-32b)
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        load_dotenv()
        self.provider = (provider or os.getenv("LLM_PROVIDER") or "stub").lower()
        self.model = model or os.getenv("LLM_MODEL") or "qwen/qwen3-32b"
        self.temperature = float(os.getenv("LLM_TEMPERATURE", str(temperature)))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", str(max_tokens)))
        self._client = None

        if self.provider == "groq":
            if Groq is None:
                raise ImportError("groq package not installed. Run: pip install groq")
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError("GROQ_API_KEY missing in environment/.env")
            self._client = Groq(api_key=api_key)
        elif self.provider == "stub":
            pass
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    # ---- JSON-returning helper for your text retriever ----
    def speculate_features(self, query: str, n_features: int = 8, schema: Optional[dict] = None) -> Dict[str, Any]:
        n_features = max(3, min(int(n_features), 12))
        schema_json = json.dumps(schema or _default_schema(), ensure_ascii=False)

        system_msg = (
            "You are a feature-speculation assistant for a multimodal health-performance system. "
            "Return ONLY strict JSON matching the schema. No prose."
        )
        user_msg = (
            "Task: Given the user's query, produce 3..N predictive independent variables (features). "
            f"N={n_features}. Rules: Each feature has name (short), rationale (1–2 sentences), weight (0..1). "
            "Weights need not sum to 1. Output ONLY JSON with the schema; no extra keys.\n\n"
            f"Schema:\n{schema_json}\n\nUser query:\n{query}"
        )

        if self.provider == "stub":
            return _stub_speculate(n_features)

        resp = self._client.chat.completions.create(  # type: ignore
            model=self.model,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": user_msg}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = resp.choices[0].message.content
        return _coerce_json(content)

    # ---- Simple chat passthrough (for your existing RAG script) ----
    def chat(self, prompt: str) -> str:
        if self.provider == "stub":
            return f"[stub:{self.model}] {prompt[:200]}"
        resp = self._client.chat.completions.create(  # type: ignore
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content


if __name__ == "__main__":
    # tiny self-test
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "stub"))
    p.add_argument("--model", default=os.getenv("LLM_MODEL", "qwen/qwen3-32b"))
    p.add_argument("--query", required=True)
    p.add_argument("--n", type=int, default=6)
    args = p.parse_args()

    llm = LLMClient(provider=args.provider, model=args.model)
    out = llm.speculate_features(query=args.query, n_features=args.n)
    print(json.dumps(out, indent=2))
