"""Product wrapper — re-exports from architecture.
Product code (SDK, MCP, scripts) can keep importing from here.
Architecture code imports from app.config.settings directly."""
from app.config.settings import Settings, settings

__all__ = ["Settings", "settings"]
