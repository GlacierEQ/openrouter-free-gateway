"""
apex-gateway-verify — End-to-end deployment verification for the OpenRouter Free Gateway MCP server.

Usage:  python3 apex-gateway-verify.py

Checks:
  1. Backend code: OllamaBackend disabled, MeshLLM/Titan commented out
  2. Backend code: _post() accepts `extra` parameter
  3. Backend code: f-string error message interpolation
  4. Server code: gateway_chat defaults to verified model when empty
  5. Server code: broken models (inkling, minimax-m3, gemma-4) commented out
  6. Config: kilo.json apex-free provider has verified models
  7. Config: optimized_router_config.json has working primary_model
  8. Runtime: BackendManager has exactly 3 backends (kilo, openrouter, novita)
  9. Runtime: all 3 backends report healthy
  10. Runtime: gateway_chat MCP call returns successful response
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

GATEWAY_DIR = Path(
    "/Users/kcbflux/APEX_SYSTEM/INFRASTRUCTURE/MCP_SERVERS/openrouter-free-gateway"
)
KILO_CONFIG = Path("/Users/kcbflux/.config/kilo/config.json")
ROUTER_CONFIG = GATEWAY_DIR / "optimized_router_config.json"

BACKEND_CODE = (GATEWAY_DIR / "backends.py").read_text()
SERVER_CODE = (GATEWAY_DIR / "server.py").read_text()

failures = []
passes = []


def check(name, condition, detail=""):
    if condition:
        passes.append(name)
        print(f"  PASS: {name}")
    else:
        failures.append(name)
        print(f"  FAIL: {name}")
        if detail:
            print(f"         {detail}")


print("=" * 60)
print("APEX Gateway Verification")
print("=" * 60)

# ── Code Checks ──
print("\n[1] Backend code checks")

check(
    "OllamaBackend disabled in BackendManager",
    "# OllamaBackend()," in BACKEND_CODE
    and BACKEND_CODE.split("self.backends: List[Backend] = [")[1]
    .split("]")[0]
    .strip()
    .startswith("#"),
    "OllamaBackend should be commented out in __init__",
)

check(
    "MeshLLMBackend disabled",
    "# MeshLLMBackend()," in BACKEND_CODE,
)

check(
    "TitanBackend disabled",
    "# TitanBackend()," in BACKEND_CODE,
)

check(
    "_post accepts extra parameter",
    "extra: Optional[Dict[str, str]] = None"
    in BACKEND_CODE.split("def _post")[1].split("def _get")[0],
)

check(
    "f-string error message interpolated",
    "{last_error}"
    in BACKEND_CODE.split("ALL_BACKENDS_EXHAUSTED")[1].split("}")[0] + "}",
)

# ── Server Checks ──
print("\n[2] Server code checks")

check(
    "gateway_chat defaults to verified model when empty",
    'requested_model = "poolside/laguna-s-2.1:free"' in SERVER_CODE
    and "if not requested_model" in SERVER_CODE,
)

# Check broken models are commented out in ALL fallback chain sections of server.py
# These references should only appear in comments (prefixed with #)
lines = SERVER_CODE.split("\n")
broken_refs = [
    "thinkingmachines/inkling:free",
    "minimax/minimax-m3:free",
    "google/gemma-4-26b-a4b-it:free",
]
uncommented_broken = []
for line in lines:
    stripped = line.lstrip()
    for ref in broken_refs:
        if ref in stripped and not stripped.startswith("#"):
            uncommented_broken.append(f"line: {stripped.strip()[:80]}")

check(
    "broken models commented out in fallback chain",
    len(uncommented_broken) == 0,
    f"Uncommented broken refs: {uncommented_broken[:3]}",
)

# ── Config Checks ──
print("\n[3] Config checks")

kilo_cfg = json.loads(KILO_CONFIG.read_text())
providers = kilo_cfg.get("provider", {})
apex_free = providers.get("apex-free", {})

model_ids = (
    [m.get("id") for m in apex_free.get("models", {}).values()]
    if isinstance(apex_free.get("models"), dict)
    else []
)
check(
    "kilo.json apex-free has verified models",
    "inclusionai/ling-3.0-flash-sante:free" in model_ids
    or "poolside/laguna-s-2.1:free" in model_ids
    or "nvidia/nemotron-3.5-lightning:free" in model_ids,
    f"Found: {model_ids}",
)

check(
    "kilo.json apex-free has no broken models",
    not any("inkling" in mid or "minimax-m3" in mid for mid in model_ids),
)

check(
    "kilo.json uses env-var API key reference",
    "${OPENROUTER_API_KEY}" in json.dumps(apex_free),
    "Should use ${OPENROUTER_API_KEY} not hardcoded key",
)

router_cfg = json.loads(ROUTER_CONFIG.read_text())
check(
    "router config primary_model is verified",
    router_cfg.get("primary_model")
    in [
        "nvidia/nemotron-3.5-lightning:free",
        "poolside/laguna-s-2.1:free",
        "openrouter/free",
    ],
    f"primary_model={router_cfg.get('primary_model', 'MISSING')}",
)

check(
    "router config fallback_chain has no broken models",
    not any(
        "inkling" in m.get("id", "") or "minimax-m3" in m.get("id", "")
        for m in router_cfg.get("fallback_chain", [])
    ),
)

# ── Runtime Checks ──
print("\n[4] Runtime checks")

sys.path.insert(0, str(GATEWAY_DIR))
# Clear any cached modules
mods = [k for k in sys.modules if "backends" in k]
for m in mods:
    del sys.modules[m]

from backends import BackendManager

mgr = BackendManager()

backend_names = [b.name for b in mgr.backends]
check(
    "BackendManager has exactly 3 backends",
    backend_names == ["kilo", "openrouter", "novita"],
    f"Got: {backend_names}",
)

check(
    "No Ollama in backend list",
    "ollama" not in backend_names,
    f"Backends: {backend_names}",
)

health = mgr.health_map()
check("All 3 backends healthy", all(health.values()), f"Health: {health}")

# ── End-to-End MCP Test ──
print("\n[5] End-to-end MCP test")

proc = subprocess.Popen(
    [sys.executable, str(GATEWAY_DIR / "server.py")],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

# Initialize
proc.stdin.write(
    json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "verify", "version": "1.0"},
            },
        }
    )
    + "\n"
)
proc.stdin.flush()
proc.stdout.readline()

# gateway_chat with NO model (tests default fallback)
proc.stdin.write(
    json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "gateway_chat",
                "arguments": {
                    "prompt": "Say ONLY: E2E_OK",
                    "temperature": 0.1,
                    "max_tokens": 128,
                },
            },
        }
    )
    + "\n"
)
proc.stdin.flush()
resp = proc.stdout.readline()
r = json.loads(resp)

if r.get("result", {}).get("content"):
    text = r["result"]["content"][0].get("text", "")
    data = json.loads(text)
    check(
        "gateway_chat returns success with default model",
        data.get("status") == "success",
        f"status={data.get('status')}, msg={data.get('message', '')}",
    )
    check(
        "gateway_chat response contains expected text",
        "E2E_OK" in data.get("response", ""),
        f"response={data.get('response', '')[:100]}",
    )
else:
    check("gateway_chat returns success", False, str(r.get("error", r)))

proc.terminate()

# ── Summary ──
print("\n" + "=" * 60)
print(f"PASSED: {len(passes)} | FAILED: {len(failures)}")
print("=" * 60)
if failures:
    print("\nFailed checks:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("\nAll checks passed. Gateway is production-ready.")
    sys.exit(0)
