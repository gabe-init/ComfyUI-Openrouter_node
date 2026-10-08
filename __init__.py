from .node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


def register_routes():
    """Expose public metadata and server-side credits without returning a key."""
    try:
        from server import PromptServer
        from aiohttp import web
    except ImportError:
        return  # Standalone tests and metadata tools do not run a Comfy server.
    if not hasattr(PromptServer, "instance"):
        return
    import asyncio
    from . import openrouter_catalog
    from .node import OpenRouterNode

    @PromptServer.instance.routes.get("/openrouter/model_catalog")
    async def model_catalog(request):
        # This only schedules background work and copies the last snapshot.
        snapshot = openrouter_catalog.get_catalog(refresh=request.query.get("refresh") == "1")
        snapshot["video"] = openrouter_catalog.video_generation_models(snapshot)
        return web.json_response(snapshot)

    @PromptServer.instance.routes.get("/openrouter/credits")
    async def credits(request):
        def fetch():
            node = OpenRouterNode.__new__(OpenRouterNode)
            return node.fetch_credits("", timeout=15)
        return web.json_response({"credits": await asyncio.to_thread(fetch)})


register_routes()

