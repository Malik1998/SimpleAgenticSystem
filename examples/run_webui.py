"""Local web UI: a ChatGPT-style chat with a settings sidebar (provider, model,
temperature, max tokens, agent iterations, tools) wired to the real Agent/LLMRouter/
ToolRegistry. Defaults to an offline demo model, so it runs with no API keys.

Run: uv run python examples/run_webui.py
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import uvicorn

from simple_agentic_system.webui import create_app


def main() -> None:
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
