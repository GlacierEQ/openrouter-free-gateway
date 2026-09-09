"""
Distributed Backend Mesh for the OpenRouter Free Gateway
=========================================================
Unified inference backend abstraction. Routes chat requests across a
priority-ordered set of backends, each exposing a common chat/list/health
interface. Local and mesh backends are preferred; cloud backends are the
failover of last resort.

Backends (priority order):
  1. Ollama          — local models, zero cost, zero latency
  2. Mesh LLM        — distributed GPU mesh (localhost:9337), pools spare GPUs
  3. TITAN           — local agent framework (localhost:48420), 36 providers
  4. OpenRouter      — cloud aggregator, the original primary
  5. Novita AI       — cloud failover

The primordial-mesh-titan stack (local-first) is preferred over cloud.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GatewayBackends")


# ---------------------------------------------------------------------------
# Base backend
# ---------------------------------------------------------------------------


class Backend:
    """Abstract inference backend. Subclasses implement chat/list_models/health."""

    name: str = "base"
    base_url: str = ""
    requires_key: bool = False

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    def chat(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def list_models(self) -> List[Dict[str, Any]]:
        return []

    def health(self) -> bool:
        return False

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "User-Agent": "APEX-Gateway/2.0"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if extra:
            h.update(extra)
        return h

    def _post(
        self,
        path: str,
        payload: dict,
        timeout: int = 30,
        extra: Optional[Dict[str, str]] = None,
    ) -> dict:
        url = f"{self.base_url}{path}"
        headers = self._headers(extra)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _get(self, path: str, timeout: int = 8) -> dict:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------


class OllamaBackend(Backend):
    name = "ollama"
    base_url = "http://localhost:11434"
    requires_key = False

    def __init__(self, base_url: str = ""):
        super().__init__("")
        if base_url:
            self.base_url = base_url.rstrip("/")

    def chat(self, model, prompt, system_prompt="", temperature=0.7, max_tokens=4096):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        # Ollama native /api/chat
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            res = self._post("/api/chat", body)
            content = res.get("message", {}).get("content", "")
            return {
                "status": "success",
                "model_used": f"ollama/{model}",
                "response": content,
                "backend": self.name,
                "usage": res.get("usage", {}),
            }
        except Exception as e:
            return {"status": "error", "backend": self.name, "message": str(e)}

    def list_models(self):
        try:
            data = self._get("/api/tags")
            models = data.get("models", [])
            return [
                {
                    "id": m.get("name", ""),
                    "name": m.get("name", ""),
                    "backend": self.name,
                }
                for m in models
            ]
        except Exception:
            return []

    def health(self):
        try:
            self._get("/api/tags")
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Mesh LLM (distributed GPU mesh, localhost:9337)
# ---------------------------------------------------------------------------


class MeshLLMBackend(Backend):
    name = "mesh-llm"
    base_url = "http://localhost:9337/v1"
    requires_key = False

    def __init__(self, base_url: str = ""):
        super().__init__("")
        if base_url:
            self.base_url = base_url.rstrip("/")

    def chat(self, model, prompt, system_prompt="", temperature=0.7, max_tokens=4096):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            res = self._post("/chat/completions", payload)
            choices = res.get("choices", [])
            content = (
                choices[0].get("message", {}).get("content", "") if choices else ""
            )
            return {
                "status": "success",
                "model_used": f"mesh/{model}",
                "response": content,
                "backend": self.name,
                "usage": res.get("usage", {}),
            }
        except Exception as e:
            return {"status": "error", "backend": self.name, "message": str(e)}

    def list_models(self):
        try:
            data = self._get("/models")
            models = data.get("data", [])
            return [
                {
                    "id": m.get("id", ""),
                    "name": m.get("name", m.get("id", "")),
                    "backend": self.name,
                    "context_length": m.get("context_length", 0),
                }
                for m in models
            ]
        except Exception:
            return []

    def health(self):
        try:
            self._get("/models")
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# TITAN (local agent framework, localhost:48420)
# ---------------------------------------------------------------------------


class TitanBackend(Backend):
    name = "titan"
    base_url = "http://localhost:48420"
    requires_key = False

    def __init__(self, base_url: str = "", api_key: str = ""):
        super().__init__(api_key)
        if base_url:
            self.base_url = base_url.rstrip("/")

    def chat(self, model, prompt, system_prompt="", temperature=0.7, max_tokens=4096):
        # TITAN exposes an OpenAI-compatible /v1/chat/completions endpoint.
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            res = self._post("/v1/chat/completions", payload)
            choices = res.get("choices", [])
            content = (
                choices[0].get("message", {}).get("content", "") if choices else ""
            )
            return {
                "status": "success",
                "model_used": f"titan/{model}",
                "response": content,
                "backend": self.name,
                "usage": res.get("usage", {}),
            }
        except Exception as e:
            return {"status": "error", "backend": self.name, "message": str(e)}

    def list_models(self):
        try:
            data = self._get("/v1/models")
            models = data.get("data", [])
            return [
                {
                    "id": m.get("id", ""),
                    "name": m.get("name", m.get("id", "")),
                    "backend": self.name,
                    "context_length": m.get("context_length", 0),
                }
                for m in models
            ]
        except Exception:
            return []

    def health(self):
        try:
            # TITAN gateway root or /v1/models
            self._get("/v1/models")
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# OpenRouter (cloud)
# ---------------------------------------------------------------------------


class OpenRouterBackend(Backend):
    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    requires_key = True

    def chat(self, model, prompt, system_prompt="", temperature=0.7, max_tokens=4096):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            res = self._post(
                "/chat/completions",
                payload,
                extra={
                    "HTTP-Referer": "https://github.com/GlacierEQ/apex-cli",
                    "X-Title": "APEX Gateway",
                },
            )
            choices = res.get("choices", [])
            content = (
                choices[0].get("message", {}).get("content", "") if choices else ""
            )
            return {
                "status": "success",
                "model_used": model,
                "response": content,
                "backend": self.name,
                "usage": res.get("usage", {}),
            }
        except Exception as e:
            return {"status": "error", "backend": self.name, "message": str(e)}

    def list_models(self):
        try:
            data = self._get("/models")
            models = data.get("data", [])
            return [
                {
                    "id": m.get("id", ""),
                    "name": m.get("name", m.get("id", "")),
                    "backend": self.name,
                    "context_length": m.get("context_length", 0),
                    "pricing": m.get("pricing", {}),
                }
                for m in models
            ]
        except Exception:
            return []

    def health(self):
        if not self.api_key:
            return False
        try:
            self._get("/models")
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Novita AI (cloud failover)
# ---------------------------------------------------------------------------


class NovitaBackend(Backend):
    name = "novita"
    base_url = "https://api.novita.ai/v3/openai"
    requires_key = True

    def chat(self, model, prompt, system_prompt="", temperature=0.7, max_tokens=4096):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        novita_model = model.replace("novita/", "").replace("openrouter/", "")
        payload = {
            "model": novita_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            res = self._post("/chat/completions", payload)
            choices = res.get("choices", [])
            content = (
                choices[0].get("message", {}).get("content", "") if choices else ""
            )
            return {
                "status": "success",
                "model_used": f"novita/{novita_model}",
                "response": content,
                "backend": self.name,
                "usage": res.get("usage", {}),
            }
        except Exception as e:
            return {"status": "error", "backend": self.name, "message": str(e)}

    def list_models(self):
        return []

    def health(self):
        if not self.api_key:
            return False
        try:
            self._get("/models")
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Kilo AI Gateway (cloud aggregator, https://api.kilo.ai/api/gateway)
# ---------------------------------------------------------------------------


class KiloBackend(Backend):
    name = "kilo"
    base_url = "https://api.kilo.ai/api/gateway"
    requires_key = True

    def chat(self, model, prompt, system_prompt="", temperature=0.7, max_tokens=4096):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            res = self._post("/chat/completions", payload)
            choices = res.get("choices", [])
            content = (
                choices[0].get("message", {}).get("content", "") if choices else ""
            )
            return {
                "status": "success",
                "model_used": f"kilo/{model}",
                "response": content,
                "backend": self.name,
                "usage": res.get("usage", {}),
            }
        except Exception as e:
            return {"status": "error", "backend": self.name, "message": str(e)}

    def list_models(self):
        try:
            data = self._get("/models")
            models = data.get("data", [])
            return [
                {
                    "id": m.get("id", ""),
                    "name": m.get("name", m.get("id", "")),
                    "backend": self.name,
                    "context_length": m.get("context_length", 0),
                }
                for m in models
            ]
        except Exception:
            return []

    def health(self):
        if not self.api_key:
            return False
        try:
            self._get("/models")
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Backend manager — discovery, health, routing
# ---------------------------------------------------------------------------


def _get_openrouter_key() -> str:
    # Prefer .env files (valid key #2) over env var (which may hold a revoked key #1)
    from pathlib import Path

    for ep in [
        Path.home() / ".config" / "kilo" / ".env",
        Path.home() / ".kilo" / ".env",
        Path.home() / ".kilocode" / ".env",
        Path.home() / ".config" / "opencode" / ".env",
        Path.home() / ".env",
    ]:
        if ep.exists():
            try:
                for line in ep.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    for prefix in (
                        "OPENROUTER_API_KEY=",
                        "OPENROUTER_TERTIARY=",
                        "OPENROUTER_PRIMARY=",
                    ):
                        if line.startswith(prefix):
                            k = line.split("=", 1)[1].strip().strip("\"'")
                            if k:
                                return k
            except Exception:
                pass
    for env_var in ["OPENROUTER_API_KEY", "OPENROUTER_TERTIARY", "OPENROUTER_PRIMARY"]:
        k = os.environ.get(env_var, "").strip()
        if k:
            return k
    return ""


def _get_novita_key() -> str:
    k = os.environ.get("NOVITA_API_KEY", "").strip()
    if k:
        return k
    from pathlib import Path

    for ep in [
        Path.home() / ".config" / "opencode" / ".env",
        Path.home() / ".config" / "kilo" / ".env",
        Path.home() / ".kilo" / ".env",
        Path.home() / ".env",
    ]:
        if ep.exists():
            try:
                for line in ep.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("NOVITA_API_KEY="):
                        v = line.strip().split("=", 1)[1].strip()
                        if v:
                            return v
            except Exception:
                pass
    return ""


def _get_kilo_key() -> str:
    """Resolve the Kilo AI Gateway JWT from env or .env files."""
    for env_var in ["KILO_API_KEY", "KILO_TOKEN"]:
        k = os.environ.get(env_var, "").strip()
        if k:
            return k
    from pathlib import Path

    for ep in [
        Path.home() / ".config" / "kilo" / ".env",
        Path.home() / ".kilo" / ".env",
        Path.home() / ".config" / "opencode" / ".env",
        Path.home() / ".env",
    ]:
        if ep.exists():
            try:
                for line in ep.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    for prefix in ("KILO_API_KEY=", "KILO_TOKEN="):
                        if line.startswith(prefix):
                            k = line.split("=", 1)[1].strip().strip("\"'")
                            if k:
                                return k
            except Exception:
                pass
    return ""


class BackendManager:
    """Owns the ordered set of backends and routes chat across them.

    Priority (primordial-mesh-titan first, cloud last):
      ollama → mesh-llm → titan → openrouter → novita
    """

    def __init__(self):
        self.backends: List[Backend] = [
            # Ollama — DISABLED: api/generate returns HTTP 000 (connection refused)
            # OllamaBackend(),
            # Mesh LLM — DISABLED: localhost:9337 unreachable (HTTP 000), no service found
            # MeshLLMBackend(),
            # TITAN — DISABLED: localhost:48420 unreachable (HTTP 000), no service found
            # TitanBackend(),
            KiloBackend(api_key=_get_kilo_key()),
            OpenRouterBackend(api_key=_get_openrouter_key()),
            NovitaBackend(api_key=_get_novita_key()),
        ]

    def health_map(self) -> Dict[str, bool]:
        return {b.name: b.health() for b in self.backends}

    def list_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for b in self.backends:
            try:
                out[b.name] = b.list_models()
            except Exception:
                out[b.name] = []
        return out

    def chat(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        preferred_backend: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Route a chat request. Tries the preferred backend first (if specified
        and healthy), then walks the priority order. Returns the first success."""
        order = list(self.backends)
        if preferred_backend:
            order = [b for b in order if b.name == preferred_backend] + [
                b for b in order if b.name != preferred_backend
            ]

        last_error: Optional[str] = None
        for b in order:
            res = b.chat(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if res.get("status") == "success":
                return res
            last_error = res.get("message", "unknown error")
            logger.warning("Backend %s failed for %s: %s", b.name, model, last_error)

        return {
            "status": "error",
            "error_type": "ALL_BACKENDS_EXHAUSTED",
            "message": f"All backends failed. Last error: {last_error}",
        }
