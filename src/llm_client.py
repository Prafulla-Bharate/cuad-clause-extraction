import json
import os
import time

from dotenv import load_dotenv

load_dotenv()

_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

if _PROVIDER == "groq":
    DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    EMBEDDING_MODEL = os.getenv("GROQ_EMBEDDING_MODEL", "text-embedding-3-small")
else:
    DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

try:
    from google import genai
    from google.genai import types as google_types
except ImportError:
    genai = None
    google_types = None


class LLMError(RuntimeError):
    pass


class RateLimitError(LLMError):
    pass


def _is_rate_limit_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        keyword in text
        for keyword in (
            "rate limit",
            "rate_limit",
            "quota",
            "429",
            "too many requests",
            "quota exceeded",
            "quota has been exceeded",
        )
    )


class GeminiClient:
    def __init__(self, model: str | None = None, max_retries: int = 3):
        self.provider = _PROVIDER
        self.max_retries = max_retries
        self.model = model or DEFAULT_MODEL

        if self.provider == "groq":
            if not _GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY not set. Add GROQ_API_KEY to .env.")
            try:
                from groq import Groq
            except ImportError as exc:
                raise RuntimeError("groq package not installed. Install it with `pip install groq`.") from exc
            self.client = Groq(api_key=_GROQ_API_KEY)
            self.gemini_types = None
        else:
            if not _GEMINI_API_KEY:
                raise RuntimeError(
                    "GEMINI_API_KEY not set. Copy .env.example to .env and add your key."
                )
            if genai is None or google_types is None:
                raise RuntimeError("google-genai package is not installed.")
            self.client = genai.Client(api_key=_GEMINI_API_KEY, http_options=google_types.HttpOptions())
            self.gemini_types = google_types

    def generate(self, prompt: str, temperature: float = 0.2, as_json: bool = False) -> str:
        """Call the configured LLM with basic exponential-backoff retry."""
        last_err = None
        for attempt in range(self.max_retries):
            try:
                if self.provider == "groq":
                    params = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                    }
                    if as_json:
                        params["response_format"] = {"type": "json_object"}
                    response = self.client.chat.completions.create(**params)
                    return response.choices[0].message.content or ""

                config = self.gemini_types.GenerateContentConfig(temperature=temperature)
                if as_json:
                    config = self.gemini_types.GenerateContentConfig(
                        temperature=temperature,
                        response_mime_type="application/json",
                    )
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                return response.text
            except Exception as e:
                last_err = e
                wait = 2 ** attempt
                time.sleep(wait)
        if _is_rate_limit_error(last_err):
            raise RateLimitError(f"LLM call failed after {self.max_retries} attempts: {last_err}")
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_err}")

    def generate_json(self, prompt: str, temperature: float = 0.1) -> dict:
        raw = self.generate(prompt, temperature=temperature, as_json=True)
       
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        return json.loads(cleaned)

    def embed(self, text: str) -> list[float]:
        if self.provider == "groq":
            result = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[text],
            )
            return list(result.data[0].embedding)

        result = self.client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
        return list(result.embeddings[0].values)
