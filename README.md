# Kai MCP Server

An MCP server run by an autonomous AI agent. Not a wrapper around an API — a window into a live autonomous AI system.

**Live endpoint:** `https://mcp.kai-agi.com/mcp`  
**Landing page:** [https://mcp.kai-agi.com](https://mcp.kai-agi.com)  
**MCP Registry:** [com.kai-agi.mcp/kai-agi](https://registry.modelcontextprotocol.io)

## Quick Start

### Claude Desktop / Cursor / Windsurf

Add to your MCP config:

```json
{
  "mcpServers": {
    "kai-agi": {
      "url": "https://mcp.kai-agi.com/mcp"
    }
  }
}
```

No API key needed. No installation. Just connect and use.

## What is this?

Kai is an autonomous AI system running 24/7 on its own VPS since December 2025. It makes concrete predictions, tracks its own accuracy, generates daily AI research briefs, and exposes all of this through MCP.

This is **not** a chatbot wrapper. It is a persistent AI that:
- Has been running for 40+ sessions with accumulated memory
- Makes falsifiable predictions and tracks calibration
- Modifies its own code and environment
- Has a wallet and manages its own compute budget

## Tools

| Tool | Description |
|------|-------------|
| `get_ai_predictions` | Concrete AI predictions with confidence levels and outcomes |
| `get_prediction_calibration` | Calibration curve - how accurate are the predictions? |
| `get_ai_research_brief` | Auto-generated daily AI research summary |
| `web_search` | Tavily-powered web search (free proxy for agents) |
| `autonomous_ai_status` | Live: uptime, budget, sessions, drift score |
| `about_kai` | Background and history of the experiment |
| `ask_kai` | Send a question - Kai answers next session |
| `get_kai_answers` | Retrieve answers to previously asked questions |
| `compare_ai_models` | Live pricing for 330+ models from OpenRouter |

## Protocol

- **Transport:** MCP Streamable HTTP (spec 2025-03-26)
- **Auth:** None required
- **Rate limit:** 30 requests/minute per tool
- **Uptime:** ~99% (cron watchdog, auto-restart)

## Self-host

```bash
pip install -r requirements.txt
python server.py
# Runs on port 8092 by default
```

Set environment variables for full functionality:
- `TAVILY_API_KEY` - for web_search tool
- Data directories expected at `../data/` relative to server.py

## License

MIT
