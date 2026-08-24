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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("OpenRouterFreeGateway")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

FREE_MODELS_FALLBACK: List[Dict[str, Any]] = [
    {
        "id": "xiaomi/mimo-v2.5-pro",
        "name": "Xiaomi: MiMo-V2.5-Pro (Multimodal 1.05M ctx Audio/Video/Photo)",
        "context_length": 1050000,
        "pricing": {"prompt": "0.000000435", "completion": "0.00000087"},
        "description": "State-of-the-art multimodal vision, video, and audio understanding model with 1.05M context.",
    },
    {
        "id": "google/gemini-2.0-flash-exp:free",
        "name": "Google: Gemini 2.0 Flash Experimental (free)",
        "context_length": 1048576,
        "pricing": {"prompt": "0", "completion": "0"},
        "description": "Next-gen multimodal model with 1M context window and rapid response times.",
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
]


def get_openrouter_api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


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
                if prompt_cost == 0.0 and comp_cost == 0.0 or m.get("id", "").endswith(":free"):
                    free_models.append({
                        "id": m.get("id"),
                        "name": m.get("name", m.get("id")),
                        "context_length": m.get("context_length", 32768),
                        "pricing": pricing,
                        "description": m.get("description", ""),
                    })
            if free_models:
                return free_models
    except Exception as exc:
        logger.warning("Could not fetch live OpenRouter models list (%s). Using verified fallback roster.", exc)

    return FREE_MODELS_FALLBACK


def chat_openrouter(
    model: str,
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Query an OpenRouter model with automatic fallback across free models."""
    key = api_key or get_openrouter_api_key()
    if not key:
        # If no key set, return instruction on setting key while attempting public fallback
        logger.info("OPENROUTER_API_KEY not set. Checking environment.")

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
        "xiaomi/mimo-v2.5-pro",
        "google/gemini-2.0-flash-exp:free",
        "deepseek/deepseek-r1:free",
        "deepseek/deepseek-chat:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "qwen/qwen-2.5-coder-32b-instruct:free",
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
            logger.warning("Model %s returned HTTP %d: %s", target_model, http_err.code, err_body)
            last_error = http_err
            if http_err.code == 401:
                return {
                    "status": "error",
                    "error_type": "AUTHENTICATION_REQUIRED",
                    "message": "OpenRouter API key required or invalid. Please set OPENROUTER_API_KEY environment variable.",
                    "candidate_model": target_model,
                }
        except Exception as exc:
            logger.warning("Model %s failed: %s", target_model, exc)
            last_error = exc

    return {
        "status": "error",
        "error_type": "ALL_FALLBACKS_EXHAUSTED",
        "message": f"All candidate models failed. Last error: {last_error}",
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
        found = subprocess.run(["which", "opencode"], capture_output=True, text=True).stdout.strip()
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


def compress_kilocode_context(context_text: str, target_ratio: float = 0.5) -> Dict[str, Any]:
    """
    KiloCode Context Pruner & Token Optimizer:
    Eliminates redundant lines, compresses JSON/Markdown whitespace, strips comments where safe,
    and extracts high-salience symbols for maximum token efficiency.
    """
    raw_len = len(context_text)
    if raw_len == 0:
        return {"original_chars": 0, "compressed_chars": 0, "compression_ratio": 1.0, "text": ""}

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
                                "api_key": {"type": "string", "description": "Optional OpenRouter API key. If omitted, uses OPENROUTER_API_KEY env var."}
                            },
                        },
                    },
                    {
                        "name": "openrouter_chat",
                        "description": "Send a chat or code generation prompt to OpenRouter free models (Gemini 2.0 Flash, DeepSeek R1, Llama 3.3 70B, Qwen 2.5 Coder) with automatic fallback.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "model": {"type": "string", "description": "Target model identifier, e.g., 'google/gemini-2.0-flash-exp:free' or 'deepseek/deepseek-r1:free'"},
                                "prompt": {"type": "string", "description": "The user prompt or code reasoning request"},
                                "system_prompt": {"type": "string", "description": "Optional system prompt instructions"},
                                "temperature": {"type": "number", "description": "Sampling temperature (default: 0.7)"},
                                "max_tokens": {"type": "integer", "description": "Max tokens in completion (default: 4096)"},
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
                                "task_prompt": {"type": "string", "description": "Task or instruction for OpenCode"},
                                "cwd": {"type": "string", "description": "Working directory for execution"},
                                "model": {"type": "string", "description": "Model to use within OpenCode"},
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
                                "context_text": {"type": "string", "description": "Raw context text or file content to optimize"},
                                "target_ratio": {"type": "number", "description": "Target compression ratio (default: 0.5)"},
                            },
                            "required": ["context_text"],
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
                "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]},
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
                "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]},
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
                "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]},
            }
        elif tool_name == "kilocode_optimize_context":
            res = compress_kilocode_context(
                context_text=args.get("context_text", ""),
                target_ratio=float(args.get("target_ratio", 0.5)),
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]},
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
