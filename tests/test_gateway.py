from __future__ import annotations

import json
import sys
from pathlib import Path

GATEWAY_DIR = Path(__file__).resolve().parents[1]
if str(GATEWAY_DIR) not in sys.path:
    sys.path.insert(0, str(GATEWAY_DIR))

from server import (
    compress_kilocode_context,
    handle_jsonrpc,
    list_free_models,
)


def test_list_free_models():
    models = list_free_models()
    assert len(models) >= 7
    ids = [m["id"] for m in models]
    assert any(":free" in m_id or "/" in m_id for m_id in ids)


def test_kilocode_context_compressor():
    sample = "unique line 1\n" + "duplicate repetitive log warning\n" * 10 + "unique line 2\n"
    res = compress_kilocode_context(sample)
    assert res["compressed_chars"] < res["original_chars"]
    assert res["compression_ratio"] < 1.0
    assert "unique line 1" in res["text"]
    assert "unique line 2" in res["text"]


def test_mcp_protocol_initialize():
    req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    resp = handle_jsonrpc(req)
    assert resp["id"] == 1
    assert resp["result"]["serverInfo"]["name"] == "openrouter-free-gateway"


def test_mcp_protocol_tools_list():
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    resp = handle_jsonrpc(req)
    assert resp["id"] == 2
    tools = resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "openrouter_list_free_models" in tool_names
    assert "openrouter_chat" in tool_names
    assert "opencode_zen_dispatch" in tool_names
    assert "kilocode_optimize_context" in tool_names


def test_mcp_tool_call_list_models():
    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "openrouter_list_free_models", "arguments": {}},
    }
    resp = handle_jsonrpc(req)
    assert resp["id"] == 3
    content = resp["result"]["content"][0]["text"]
    parsed = json.loads(content)
    assert len(parsed) >= 7


def test_mcp_tool_call_kilocode_compress():
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "kilocode_optimize_context",
            "arguments": {"context_text": "hello\n\n\nworld\n"},
        },
    }
    resp = handle_jsonrpc(req)
    assert resp["id"] == 4
    content = json.loads(resp["result"]["content"][0]["text"])
    assert "text" in content
