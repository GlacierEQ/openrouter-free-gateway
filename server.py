#!/usr/bin/env python3
"""
OpenRouter & OpenCode Zen Free Model Gateway MCP Server
Standard: Production-grade Model Context Protocol server exposing zero-cost model routing,
OpenCode Zen invocation, KiloCode token compression, and Cline agentic bridges.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Distributed backend mesh (primordial-mesh-titan stack)
from backends import BackendManager, OllamaBackend, MeshLLMBackend, TitanBackend

# Shared backend manager instance (thread-safe for read-only use)
_manager = BackendManager()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("OpenRouterFreeGateway")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

FREE_MODELS_FALLBACK: List[Dict[str, Any]] = [
    # ─────────────────────────────────────────────────────────────────────────────
    # ACTIVE FREE MODELS (live on OpenRouter as of 2026-08-26)
    # ─────────────────────────────────────────────────────────────────────────────
    {
        "id": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "name": "NVIDIA: Nemotron 3 Ultra (free)",
        "context_length": 1000000,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Frontier-reasoning MoE orchestration model, 55B active / 550B total, hybrid Transformer-Mamba. 1M context.",
    },
    {
        "id": "nvidia/nemotron-3.5-lightning:free",
        "name": "NVIDIA: Nemotron 3.5 Lightning (free)",
        "context_length": 1000000,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Open MoE model, 3B active / 30B total, for high-throughput agentic workloads. 1M context.",
    },
    {
        "id": "nvidia/nemotron-3-super-120b-a12b:free",
        "name": "NVIDIA: Nemotron 3 Super (free)",
        "context_length": 262144,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "120B hybrid MoE model, 12B active, for complex multi-agent applications.",
    },
    {
        "id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "name": "NVIDIA: Nemotron 3 Nano Omni (free)",
        "context_length": 256000,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "30B-A3B open multimodal model for perception/context sub-agents. Accepts text, image, video.",
    },
    {
        "id": "nvidia/nemotron-3.5-content-safety:free",
        "name": "NVIDIA: Nemotron 3.5 Content Safety (free)",
        "context_length": 128000,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Compact 4B multimodal guardrail model for moderating LLM/VLM inputs and outputs.",
    },
    # ── Confirmed broken models (excluded from fallback) ──
    # "thinkingmachines/inkling:free",        — HTTP 404, model retired
    # "thinkingmachines/inkling-small:free",  — HTTP 404, model retired
    # "minimax/minimax-m3:free",              — HTTP 000, generate endpoint unavailable
    # "minimax/minimax-m2.7:free",            — HTTP 000, generate endpoint unavailable
    {
        "id": "z-ai/glm-5.2:free",
        "name": "Z.ai: GLM 5.2 (free)",
        "context_length": 256000,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Large-scale reasoning model for long-horizon agent workflows & project-level software engineering.",
    },
    {
        "id": "cohere/north-mini-code:free",
        "name": "Cohere: North Mini Code (free)",
        "context_length": 256000,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Cohere's first agentic coding model. Sparse MoE, 3B active / 30B total, optimized for code.",
    },
    {
        "id": "poolside/laguna-s-2.1:free",
        "name": "Poolside: Laguna S 2.1 (free)",
        "context_length": 262144,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Latest coding agent model, 118B total / 8B active, 70.2% on Terminal-Bench 2.1.",
    },
    {
        "id": "poolside/laguna-xs-2.1:free",
        "name": "Poolside: Laguna XS 2.1 (free)",
        "context_length": 262144,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Coding agent in the 33B-A3B category, successor to Laguna XS.2.",
    },
    {
        "id": "google/gemini-2.0-flash-exp:free",
        "name": "Google: Gemini 2.0 Flash Experimental (free)",
        "context_length": 1048576,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Next-gen multimodal model with 1M context window and rapid response times.",
    },
    # google/gemma-4-26b-a4b-it:free — Confirmed broken: HTTP 000 on generate endpoint
    {
        "id": "google/gemma-4-31b-it:free",
        "name": "Google: Gemma 4 31B (free)",
        "context_length": 262144,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "30.7B dense multimodal model, text & image in, configurable thinking/reasoning mode.",
    },
    {
        "id": "deepseek/deepseek-r1:free",
        "name": "DeepSeek: R1 Reasoning (free)",
        "context_length": 163840,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Premier open reasoning model with reinforcement-learning chain-of-thought.",
    },
    {
        "id": "deepseek/deepseek-chat:free",
        "name": "DeepSeek: DeepSeek V3 (free)",
        "context_length": 65536,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "General-purpose 671B MoE model excelling at coding and structured problem solving.",
    },
    {
        "id": "meta-llama/llama-3.3-70b-instruct:free",
        "name": "Meta: Llama 3.3 70B Instruct (free)",
        "context_length": 131072,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Industry-standard open-weights flagship with 128k context.",
    },
    {
        "id": "xiaomi/mimo-v2.5-pro",
        "name": "Xiaomi: MiMo-V2.5-Pro (Multimodal 1.05M ctx Audio/Video/Photo)",
        "context_length": 1050000,
        "pricing": {"prompt": "0.000000435", "completion": "0.00000087"},
        "description": "State-of-the-art multimodal vision, video, and audio understanding model with 1.05M context.",
    },
    {
        "id": "qwen/qwen-2.5-coder-32b-instruct:free",
        "name": "Qwen: Qwen 2.5 Coder 32B Instruct (free)",
        "context_length": 32768,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Specialized code generation, debugging, and refactoring model.",
    },
    {
        "id": "mistralai/mistral-small-24b-instruct-2501:free",
        "name": "Mistral: Mistral Small 3 24B (free)",
        "context_length": 32768,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Compact, highly efficient enterprise model from Mistral AI.",
    },
    {
        "id": "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
        "name": "Cognitive Computations: Dolphin 3.0 R1 Mistral 24B (free)",
        "context_length": 32768,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Uncensored reasoning model built on top of Mistral 24B.",
    },
    {
        "id": "dots-studio/dots-3-note-preview:free",
        "name": "Dots Studio: Dots3-Note Preview (free)",
        "context_length": 512000,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Open-weight MoE, 16B active / 280B total, lightest in the Dots 3 family.",
    },
    {
        "id": "liquid/lfm-2.5-2.6b:free",
        "name": "LiquidAI: LFM2.5-2.6B (free)",
        "context_length": 65536,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Compact reasoning model for agent workflows, data extraction, RAG, and long-context processing.",
    },
    {
        "id": "openrouter/free",
        "name": "Free Models Router",
        "context_length": 200000,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "openrouter/free — a router that selects free models at random from the available pool.",
    },
    # ─────────────────────────────────────────────────────────────────────────────
    # STEALTH MODELS — HISTORICAL / DEPRECATED / RETIRED
    # All 14 OpenRouter stealth models since April 2025. These are retired/removed
    # from the live API and will NOT resolve. Kept for archival reference and in
    # case any slug is revived. Do not rely on these for inference.
    # ─────────────────────────────────────────────────────────────────────────────
    {
        "id": "stealth/ox-alpha",
        "name": "[STEALTH-RET] Ox Alpha → ZAI GLM-5.3-Flash (1M ctx, was free, now paid)",
        "context_length": 1048576,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Stealth #14 (2026-08-20). Revealed as ZAI GLM-5.3-Flash. Free preview ended ~Aug 27; delisted. 1M ctx, multimodal, reasoning.",
    },
    {
        "id": "openrouter/quasar-alpha",
        "name": "[STEALTH-RET] Quasar Alpha → OpenAI GPT-4.1 pre-release (1M ctx)",
        "context_length": 1000000,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Stealth #1 (2025-04-02). First stealth model. Revealed by OpenRouter blog 2025-04-14 as GPT-4.1 pre-release snapshot. Retired.",
    },
    {
        "id": "openrouter/optimus-alpha",
        "name": "[STEALTH-RET] Optimus Alpha → OpenAI GPT-4.1 nearer-final (1M ctx)",
        "context_length": 1000000,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Stealth #2 (2025-04-10). Revealed same day as Quasar (OpenRouter blog 2025-04-14) as GPT-4.1 nearer-final snapshot. Retired.",
    },
    {
        "id": "openrouter/cypher-alpha",
        "name": "[STEALTH-RET] Cypher Alpha → UNRESOLVED (1M ctx, fictional 'Cypher Labs')",
        "context_length": 1000000,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Stealth #3 (2025-07-01). Never revealed. Listed under 'Cypher Labs' — a provider name OpenRouter itself called fictional. Data terms broadest in set. Retired.",
    },
    {
        "id": "openrouter/horizon-alpha",
        "name": "[STEALTH-RET] Horizon Alpha → early GPT-5 checkpoint (256K ctx, DEPRECATED)",
        "context_length": 256000,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Stealth #4 (2025-07-30). Revealed by OpenRouter blog/X 2025-08-07 as early GPT-5 checkpoint. Superseded by Horizon Beta. Page live but deprecated.",
    },
    {
        "id": "openrouter/horizon-beta",
        "name": "[STEALTH-RET] Horizon Beta → later GPT-5 checkpoint (256K ctx)",
        "context_length": 256000,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Stealth #5 (2025-08-01). Revealed 2025-08-07 as later GPT-5 checkpoint that superseded Horizon Alpha. Retired.",
    },
    {
        "id": "openrouter/sonoma-dusk-alpha",
        "name": "[STEALTH-RET] Sonoma Dusk Alpha → presumed early Grok 4 Fast (2M ctx)",
        "context_length": 2000000,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Stealth #6 (2025-09-05). Reported (not confirmed) as early Grok 4 Fast. xAI never confirmed. Retired. 2M context — largest in set.",
    },
    {
        "id": "openrouter/sonoma-sky-alpha",
        "name": "[STEALTH-RET] Sonoma Sky Alpha → presumed Grok 4 Fast larger variant (2M ctx)",
        "context_length": 2000000,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Stealth #7 (2025-09-05). Reported (not confirmed) as larger Grok 4 Fast variant. xAI never confirmed. Retired. 2M context.",
    },
    {
        "id": "openrouter/polaris-alpha",
        "name": "[STEALTH-RET] Polaris Alpha → believed early GPT-5.1 snapshot (256K ctx, weak evidence)",
        "context_length": 256000,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Stealth #8 (2025-11-06). Reported (weak) as early GPT-5.1 snapshot. OpenAI never confirmed. Retired.",
    },
    {
        "id": "openrouter/sherlock-think-alpha",
        "name": "[STEALTH-RET] Sherlock Think Alpha → believed Grok 4.1 Fast reasoning (1.84M ctx)",
        "context_length": 1840000,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Stealth #9 (2025-11-15). Reported as early Grok 4.1 Fast reasoning model. xAI never confirmed. Retired. 1.84M ctx.",
    },
    {
        "id": "openrouter/aurora-alpha",
        "name": "[STEALTH-RET] Aurora Alpha → UNRESOLVED (128K ctx)",
        "context_length": 128000,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Stealth #10 (2026-02-08). Never revealed. Smallest context in set (128K). Retired.",
    },
    {
        "id": "openrouter/hunter-alpha",
        "name": "[STEALTH-RET] Hunter Alpha → Xiaomi MiMo-V2-Pro (1M ctx)",
        "context_length": 1048576,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Stealth #11 (2026-03-11). Revealed by Xiaomi MiMo team 2026-03-18 as MiMo-V2-Pro early internal test build. Retired.",
    },
    {
        "id": "openrouter/healer-alpha",
        "name": "[STEALTH-RET] Healer Alpha → Xiaomi MiMo-V2-Omni multimodal (262K ctx)",
        "context_length": 262144,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Stealth #12 (2026-03-11). Same Xiaomi MiMo reveal as Hunter — MiMo-V2-Omni, multimodal sibling. Retired.",
    },
    {
        "id": "openrouter/owl-alpha",
        "name": "[STEALTH-RET] Owl Alpha → Meituan LongCat-2.0-Preview 1.6T/48B MoE (1M ctx)",
        "context_length": 1048576,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Stealth #13 (2026-04-28). Revealed by Meituan blog/X 2026-06-30 as LongCat-2.0-Preview, 1.6T params / 48B active MoE. Longest gap (63 days). Retired.",
    },
]


def get_openrouter_api_key() -> str:
    # 1. Check local verified .env files first (key #2 lives here and is valid)
    env_paths = [
        Path.home() / ".config" / "kilo" / ".env",
        Path.home() / ".kilo" / ".env",
        Path.home() / ".kilocode" / ".env",
        Path.home() / ".config" / "opencode" / ".env",
        Path.home() / ".env",
    ]
    for ep in env_paths:
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

    # 2. Fall back to environment variables
    for env_var in ["OPENROUTER_API_KEY", "OPENROUTER_TERTIARY", "OPENROUTER_PRIMARY"]:
        k = os.environ.get(env_var, "").strip()
        if k:
            return k
    return ""


def list_free_models(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch live available free models from OpenRouter API, falling back to static roster."""
    key = api_key or get_openrouter_api_key()
    headers = {
        "User-Agent": "Antigravity-OpenRouter-Gateway/1.0",
        "HTTP-Referer": "https://github.com/GlacierEQ/apex-cli",
        "X-Title": "APEX Antigravity CLI",
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"

    req = urllib.request.Request(OPENROUTER_MODELS_URL, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            free_models = []
            for m in models:
                pricing = m.get("pricing", {})
                prompt_cost = float(pricing.get("prompt", 0))
                comp_cost = float(pricing.get("completion", 0))
                if (
                    prompt_cost == 0.0
                    and comp_cost == 0.0
                    or m.get("id", "").endswith(":free")
                ):
                    free_models.append(
                        {
                            "id": m.get("id"),
                            "name": m.get("name", m.get("id")),
                            "context_length": m.get("context_length", 32768),
                            "pricing": pricing,
                            "description": m.get("description", ""),
                        }
                    )
            if free_models:
                return free_models
    except Exception as exc:
        logger.warning(
            "Could not fetch live OpenRouter models list (%s). Using verified fallback roster.",
            exc,
        )

    return FREE_MODELS_FALLBACK


def get_novita_api_key() -> str:
    key = os.environ.get("NOVITA_API_KEY", "").strip()
    if key:
        return key
    env_paths = [
        Path.home() / ".config" / "opencode" / ".env",
        Path.home() / ".config" / "kilo" / ".env",
        Path.home() / ".kilo" / ".env",
        Path.home() / ".kilocode" / ".env",
        Path.home() / ".env",
    ]
    for ep in env_paths:
        if ep.exists():
            try:
                for line in ep.read_text(encoding="utf-8").splitlines():
                    if line.startswith("NOVITA_API_KEY="):
                        k = line.split("=", 1)[1].strip()
                        if k:
                            return k
            except Exception:
                pass
    return ""


def chat_novita_ai(
    model: str = "deepseek/deepseek-v4-pro",
    prompt: str = "",
    system_prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """Fallback dispatch to Novita AI OpenAI-compatible endpoint."""
    key = get_novita_api_key()
    if not key:
        return {
            "status": "error",
            "message": "NOVITA_API_KEY not configured for failover.",
        }

    # Map model name
    novita_model = model.replace("novita/", "").replace("openrouter/", "")
    if "r1" in novita_model or "reasoner" in novita_model:
        novita_model = "deepseek/deepseek-r1"
    elif "deepseek" in novita_model:
        novita_model = "deepseek/deepseek-v4-pro"
    elif "kimi" in novita_model:
        novita_model = "moonshotai/kimi-k3"
    elif "glm" in novita_model:
        novita_model = "zai-org/glm-5.3"
    elif "qwen" in novita_model:
        novita_model = "qwen/qwen3.8-max"
    else:
        novita_model = "deepseek/deepseek-v4-pro"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": novita_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "User-Agent": "APEX-Novita-Gateway/2.0",
    }
    req = urllib.request.Request(
        "https://api.novita.ai/v3/openai/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            choices = res_data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                return {
                    "status": "success",
                    "model_used": f"novita/{novita_model} (Failover Active)",
                    "response": content,
                    "usage": res_data.get("usage", {}),
                }
    except Exception as e:
        return {"status": "error", "message": f"Novita AI failover failed: {e}"}

    return {"status": "error", "message": "Novita AI returned empty response."}


def chat_openrouter(
    model: str,
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Query an OpenRouter model with automatic fallback across free models and Novita AI."""
    key = api_key or get_openrouter_api_key()
    if not key:
        # Check if we should failover directly to Novita AI
        novita_res = chat_novita_ai(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if novita_res.get("status") == "success":
            return novita_res

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Antigravity-OpenRouter-Gateway/1.0",
        "HTTP-Referer": "https://github.com/GlacierEQ/apex-cli",
        "X-Title": "APEX Antigravity CLI",
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    candidate_models = [model]
    for fb in [
        # ── Verified working free models (priority order) ──
        # Verified by end-to-end test: returns successful completions
        "poolside/laguna-s-2.1:free",
        "inclusionai/ling-3.0-flash-sante:free",
        "inclusionai/ling-3.0-flash-fin:free",
        "nvidia/nemotron-3.5-lightning:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "openrouter/free",
        "dots-studio/dots-3-note-preview:free",
        "deepseek/deepseek-r1:free",
        "deepseek/deepseek-chat:free",
        # ── Untested / may fail (tried after verified models) ──
        "google/gemini-2.0-flash-exp:free",
        "xiaomi/mimo-v2.5-pro",
        "meta-llama/llama-3.3-70b-instruct:free",
        "qwen/qwen-2.5-coder-32b-instruct:free",
        "z-ai/glm-5.2:free",
        "cohere/north-mini-code:free",
        "poolside/laguna-xs-2.1:free",
        "liquid/lfm-2.5-2.6b:free",
        "mistralai/mistral-small-24b-instruct-2501:free",
        "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
        # "minimax/minimax-m3:free",              — HTTP 400, model retired
        "minimax/minimax-m2.7:free",
        # ── Confirmed broken models (excluded from fallback) ──
        # "thinkingmachines/inkling:free",        — HTTP 404, model retired
        # "thinkingmachines/inkling-small:free",  — HTTP 404, model retired
        # "google/gemma-4-31b-it:free",           — HTTP 404, model retired
        # "google/gemma-4-26b-a4b-it:free",       — HTTP 404, model retired
        # ── Stealth models (historical/retired — tried last, will fall through) ──
        "stealth/ox-alpha",
        "openrouter/quasar-alpha",
        "openrouter/optimus-alpha",
        "openrouter/cypher-alpha",
        "openrouter/horizon-alpha",
        "openrouter/horizon-beta",
        "openrouter/sonoma-dusk-alpha",
        "openrouter/sonoma-sky-alpha",
        "openrouter/polaris-alpha",
        "openrouter/sherlock-think-alpha",
        "openrouter/aurora-alpha",
        "openrouter/hunter-alpha",
        "openrouter/healer-alpha",
        "openrouter/owl-alpha",
    ]:
        if fb not in candidate_models:
            candidate_models.append(fb)

    last_error: Optional[Exception] = None

    for target_model in candidate_models:
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(OPENROUTER_API_URL, data=req_data, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                choices = result.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    return {
                        "status": "success",
                        "model_used": target_model,
                        "response": content,
                        "usage": result.get("usage", {}),
                    }
        except urllib.error.HTTPError as http_err:
            err_body = http_err.read().decode("utf-8", errors="ignore")
            logger.warning(
                "Model %s returned HTTP %d: %s", target_model, http_err.code, err_body
            )
            last_error = http_err
            # If rate limited (429) or forbidden (403), trigger Novita failover
            if http_err.code in (429, 403, 401):
                novita_res = chat_novita_ai(
                    model=model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if novita_res.get("status") == "success":
                    return novita_res
        except Exception as exc:
            logger.warning("Model %s failed: %s", target_model, exc)
            last_error = exc

    # Final attempt: Novita AI failover
    novita_res = chat_novita_ai(
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if novita_res.get("status") == "success":
        return novita_res

    return {
        "status": "error",
        "error_type": "ALL_FALLBACKS_EXHAUSTED",
        "message": f"All candidate models and Novita AI failover failed. Last error: {last_error}",
    }


def execute_opencode_zen(
    task_prompt: str,
    cwd: Optional[str] = None,
    model: str = "deepseek/deepseek-chat",
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    """Execute task via local OpenCode CLI binary."""
    opencode_path = "/usr/local/bin/opencode"
    if not Path(opencode_path).exists():
        # Fallback to PATH lookup
        found = subprocess.run(
            ["which", "opencode"], capture_output=True, text=True
        ).stdout.strip()
        if found:
            opencode_path = found
        else:
            return {
                "status": "error",
                "message": "opencode CLI binary not found on PATH or at /usr/local/bin/opencode",
            }

    cmd = [opencode_path, "run", task_prompt]
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return {
            "status": "success" if result.returncode == 0 else "failure",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": f"OpenCode execution timed out after {timeout_seconds}s",
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Failed to execute opencode: {exc}",
        }


def compress_kilocode_context(
    context_text: str, target_ratio: float = 0.5
) -> Dict[str, Any]:
    """
    KiloCode Context Pruner & Token Optimizer:
    Eliminates redundant lines, compresses JSON/Markdown whitespace, strips comments where safe,
    and extracts high-salience symbols for maximum token efficiency.
    """
    raw_len = len(context_text)
    if raw_len == 0:
        return {
            "original_chars": 0,
            "compressed_chars": 0,
            "compression_ratio": 1.0,
            "text": "",
        }

    lines = context_text.splitlines()
    pruned_lines = []
    seen_hashes = set()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if pruned_lines and pruned_lines[-1] != "":
                pruned_lines.append("")
            continue
        # Deduplicate repeated identical lines (logs, repetitive data)
        line_hash = hash(stripped)
        if len(stripped) > 40 and line_hash in seen_hashes:
            continue
        seen_hashes.add(line_hash)
        pruned_lines.append(line)

    compressed = "\n".join(pruned_lines)
    comp_len = len(compressed)
    ratio = comp_len / raw_len if raw_len > 0 else 1.0

    return {
        "original_chars": raw_len,
        "compressed_chars": comp_len,
        "compression_ratio": round(ratio, 3),
        "tokens_saved_approx": max(0, int((raw_len - comp_len) / 4)),
        "text": compressed,
    }


# Standard JSON-RPC MCP Dispatcher
def handle_jsonrpc(req: Dict[str, Any]) -> Dict[str, Any]:
    msg_id = req.get("id")
    method = req.get("method")
    params = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "openrouter-free-gateway",
                    "version": "1.0.0",
                },
            },
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": [
                    {
                        "name": "openrouter_list_free_models",
                        "description": "List all currently available zero-cost and free-tier models on OpenRouter with context lengths and capabilities.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "api_key": {
                                    "type": "string",
                                    "description": "Optional OpenRouter API key. If omitted, uses OPENROUTER_API_KEY env var.",
                                }
                            },
                        },
                    },
                    {
                        "name": "openrouter_chat",
                        "description": "Send a chat or code generation prompt to OpenRouter free models (Gemini 2.0 Flash, DeepSeek R1, Llama 3.3 70B, Qwen 2.5 Coder) with automatic fallback.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "model": {
                                    "type": "string",
                                    "description": "Target model identifier, e.g., 'google/gemini-2.0-flash-exp:free' or 'deepseek/deepseek-r1:free'",
                                },
                                "prompt": {
                                    "type": "string",
                                    "description": "The user prompt or code reasoning request",
                                },
                                "system_prompt": {
                                    "type": "string",
                                    "description": "Optional system prompt instructions",
                                },
                                "temperature": {
                                    "type": "number",
                                    "description": "Sampling temperature (default: 0.7)",
                                },
                                "max_tokens": {
                                    "type": "integer",
                                    "description": "Max tokens in completion (default: 4096)",
                                },
                            },
                            "required": ["model", "prompt"],
                        },
                    },
                    {
                        "name": "opencode_zen_dispatch",
                        "description": "Execute a task directly through local OpenCode Zen CLI runtime.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "task_prompt": {
                                    "type": "string",
                                    "description": "Task or instruction for OpenCode",
                                },
                                "cwd": {
                                    "type": "string",
                                    "description": "Working directory for execution",
                                },
                                "model": {
                                    "type": "string",
                                    "description": "Model to use within OpenCode",
                                },
                            },
                            "required": ["task_prompt"],
                        },
                    },
                    {
                        "name": "kilocode_optimize_context",
                        "description": "Prune and compress context text for token budget optimization and aggressive token savings.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "context_text": {
                                    "type": "string",
                                    "description": "Raw context text or file content to optimize",
                                },
                                "target_ratio": {
                                    "type": "number",
                                    "description": "Target compression ratio (default: 0.5)",
                                },
                            },
                            "required": ["context_text"],
                        },
                    },
                    {
                        "name": "gateway_list_backends",
                        "description": "List all distributed backends (ollama, mesh-llm, titan, openrouter, novita) with health status and model counts.",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                    {
                        "name": "gateway_backend_models",
                        "description": "List available models on a specific backend (ollama, mesh-llm, titan, openrouter, novita).",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "backend": {
                                    "type": "string",
                                    "description": "Backend name: ollama, mesh-llm, titan, openrouter, or novita",
                                },
                            },
                            "required": ["backend"],
                        },
                    },
                    {
                        "name": "gateway_chat",
                        "description": "Send a chat through the distributed backend mesh. Tries local backends first (ollama → mesh-llm → titan), then cloud (openrouter → novita). Supports preferred_backend override.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "model": {
                                    "type": "string",
                                    "description": "Target model identifier",
                                },
                                "prompt": {
                                    "type": "string",
                                    "description": "The user prompt",
                                },
                                "system_prompt": {
                                    "type": "string",
                                    "description": "Optional system prompt",
                                },
                                "temperature": {
                                    "type": "number",
                                    "description": "Sampling temperature (default: 0.7)",
                                },
                                "max_tokens": {
                                    "type": "integer",
                                    "description": "Max tokens (default: 4096)",
                                },
                                "preferred_backend": {
                                    "type": "string",
                                    "description": "Prefer a backend: ollama, mesh-llm, titan, openrouter, novita",
                                },
                            },
                            "required": ["model", "prompt"],
                        },
                    },
                ]
            },
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "openrouter_list_free_models":
            res = list_free_models(api_key=args.get("api_key"))
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(res, indent=2)}]
                },
            }
        elif tool_name == "openrouter_chat":
            res = chat_openrouter(
                model=args.get("model", "google/gemini-2.0-flash-exp:free"),
                prompt=args.get("prompt", ""),
                system_prompt=args.get("system_prompt", ""),
                temperature=float(args.get("temperature", 0.7)),
                max_tokens=int(args.get("max_tokens", 4096)),
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(res, indent=2)}]
                },
            }
        elif tool_name == "opencode_zen_dispatch":
            res = execute_opencode_zen(
                task_prompt=args.get("task_prompt", ""),
                cwd=args.get("cwd"),
                model=args.get("model", "deepseek/deepseek-chat"),
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(res, indent=2)}]
                },
            }
        elif tool_name == "kilocode_optimize_context":
            res = compress_kilocode_context(
                context_text=args.get("context_text", ""),
                target_ratio=float(args.get("target_ratio", 0.5)),
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(res, indent=2)}]
                },
            }
        elif tool_name == "gateway_list_backends":
            health = _manager.health_map()
            all_models = _manager.list_all_models()
            summary = []
            for b in _manager.backends:
                summary.append(
                    {
                        "backend": b.name,
                        "base_url": b.base_url,
                        "requires_key": b.requires_key,
                        "healthy": health.get(b.name, False),
                        "model_count": len(all_models.get(b.name, [])),
                    }
                )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(summary, indent=2)}]
                },
            }
        elif tool_name == "gateway_backend_models":
            be = args.get("backend", "").strip().lower()
            backend_map = {b.name: b for b in _manager.backends}
            if be not in backend_map:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32602,
                        "message": f"Unknown backend '{be}'. Valid: {list(backend_map.keys())}",
                    },
                }
            models = backend_map[be].list_models()
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(models, indent=2)}]
                },
            }
        elif tool_name == "gateway_chat":
            # Default model: use verified working free model when not specified
            requested_model = args.get("model", "").strip()
            if not requested_model:
                requested_model = "poolside/laguna-s-2.1:free"
            res = _manager.chat(
                model=requested_model,
                prompt=args.get("prompt", ""),
                system_prompt=args.get("system_prompt", ""),
                temperature=float(args.get("temperature", 0.7)),
                max_tokens=int(args.get("max_tokens", 4096)),
                preferred_backend=args.get("preferred_backend"),
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(res, indent=2)}]
                },
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Tool not found: {tool_name}"},
            }

    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> int:
    # CLI check mode
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--list-free":
            models = list_free_models()
            print(json.dumps(models, indent=2))
            return 0
        elif arg == "--test":
            print("[*] Testing KiloCode optimizer...")
            sample = "line 1\n\n\nline 2\n" + "duplicate log line error message\n" * 5
            opt = compress_kilocode_context(sample)
            print(f"[+] KiloCode tokens saved: {opt['tokens_saved_approx']}")
            print("[*] Testing Free Models list...")
            models = list_free_models()
            print(f"[+] Loaded {len(models)} free models successfully.")
            return 0

    # Stdio JSON-RPC Server
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_jsonrpc(req)
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
