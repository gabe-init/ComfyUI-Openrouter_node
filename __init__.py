import os
from .node import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

API_KEY_FILE = os.path.join(os.path.dirname(__file__), "openrouter_api_key.txt")


def _mask_api_key(key: str) -> str:
    if not key or len(key) < 8:
        return ""
    return key[:4] + "..." + key[-4:]


def _read_api_key() -> str:
    try:
        with open(API_KEY_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except (FileNotFoundError, OSError):
        return os.environ.get("OPENROUTER_API_KEY", "").strip()


try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/openrouter/api_key_status")
    async def api_key_status(request):
        key = _read_api_key()
        if key:
            source = "file" if os.path.exists(API_KEY_FILE) else "env"
            return web.json_response({"saved": True, "source": source, "masked": _mask_api_key(key)})
        return web.json_response({"saved": False, "source": "", "masked": ""})

    @PromptServer.instance.routes.post("/openrouter/save_api_key")
    async def save_api_key(request):
        json_data = await request.json()
        key = json_data.get("api_key", "").strip()
        try:
            with open(API_KEY_FILE, "w", encoding="utf-8") as f:
                f.write(key)
            return web.json_response({"success": True, "masked": _mask_api_key(key)})
        except OSError as e:
            return web.json_response({"success": False, "error": str(e)})

    @PromptServer.instance.routes.post("/openrouter/delete_api_key")
    async def delete_api_key(request):
        try:
            if os.path.exists(API_KEY_FILE):
                os.remove(API_KEY_FILE)
            return web.json_response({"success": True})
        except OSError as e:
            return web.json_response({"success": False, "error": str(e)})

except Exception:
    pass
