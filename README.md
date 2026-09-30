# openrouter-free-gateway

> Production-grade Openrouter Free Gateway system with strict contract enforcement, receipt-backed operations, and verified test coverage.

**Domain:** Autonomous Systems Engineering
**Company Orbit:** GlacierEQ

## Architecture

```
src/openrouter_free_gateway/
├── __init__.py
└── core.py          # Autonomous Systems Engineering implementation
tests/
└── test_contracts.py
.github/workflows/
└── ci.yml           # Automated CI enforcement
```

## Quick Start

```bash
# Run tests
PYTHONPATH=src pytest tests/ -v

# Lint
ruff check src/ tests/
```

## Key Classes

| Class | Purpose |
|-------|---------|
| `ContractEnforcer` | Receipt-backed operational contract enforcement. |

## License

MIT
