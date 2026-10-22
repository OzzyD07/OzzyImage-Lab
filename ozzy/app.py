import base64
import hmac
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from mcp.server.fastmcp import FastMCP, Image as MCPImage
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field

from .engine import Engine
from .schema import Generation
from .service import Studio
from .storage import Store


def create_app(root=None, engine=None, token=None):
    token = token or os.environ.get("OZZY_TOKEN", "")
    if len(token) < 24:
        raise RuntimeError("OZZY_TOKEN en az 24 karakter olmalı.")
    studio = Studio(Store(root or os.environ.get("OZZY_DATA", "data")),
                    engine or Engine(os.environ.get("OZZY_CONFIG"), os.environ.get("OZZY_DEMO") == "1"))
    cookie_key = secrets.token_urlsafe(32)
    public_host = os.environ.get("OZZY_PUBLIC_HOST", "127.0.0.1:8000")
    mcp = FastMCP("OzzyImage Lab", stateless_http=True, json_response=True, streamable_http_path="/",
                  transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=True,
                    allowed_hosts=[public_host, "127.0.0.1:*", "localhost:*", "testserver"],
                    allowed_origins=[f"https://{public_host}", "http://127.0.0.1:*", "http://localhost:*"]))

    @mcp.tool()
    def capabilities() -> dict:
        """Read model features, readiness, memory profile and backend version."""
        return studio.engine.capabilities()

    @mcp.tool()
    def upload_image(base64_data: str) -> dict:
        """Upload PNG/JPEG/WebP as raw base64. Returns an asset ID for references or masks."""
        return studio.store.upload_base64(base64_data)

    @mcp.tool()
    def generate_image(settings: Generation) -> dict:
        """Queue generation, reference editing, masked editing, RGBA or subject extraction. Returns job ID immediately."""
        return studio.submit(settings)

    @mcp.tool()
    def get_job(job_id: str) -> dict:
        """Get job status, engine log tail, output IDs and actual seeds."""
        return studio.get(job_id)

    @mcp.tool()
    def list_jobs(limit: int = 30) -> list[dict]:
        """List recent jobs and gallery outputs."""
        return studio.list_jobs(limit)

    @mcp.tool()
    def cancel_job(job_id: str) -> dict:
        """Cancel a queued or running job, retaining already completed outputs."""
        return studio.cancel(job_id)

    @mcp.tool()
    def get_image(image_id: str, asset: bool = False) -> MCPImage:
        """Read an output or uploaded asset as MCP image content."""
        return MCPImage(path=str(studio.store.path("assets" if asset else "outputs", image_id)))

    @mcp.tool()
    def get_recipe(image_id: str) -> dict:
        """Read output metadata and the generation recipe."""
        return studio.store.read("outputs", image_id)

    @mcp.tool()
    def reuse_output(image_id: str) -> dict:
        """Copy an output into reference assets for another edit."""
        return studio.store.upload(studio.store.path("outputs", image_id).read_bytes())

    @mcp.tool()
    def save_preset(name: str, settings: Generation) -> dict:
        """Save a named generation recipe to Drive."""
        return studio.save_preset(name, settings.model_dump())

    @mcp.tool()
    def list_presets() -> list[dict]:
        """List saved generation recipes."""
        return studio.store.records("presets")

    @mcp.tool()
    def delete_preset(preset_id: str) -> dict:
        """Delete a saved recipe; generated images remain available."""
        studio.store.path("presets", preset_id, ".json").unlink()
        return {"deleted": preset_id}

    @asynccontextmanager
    async def lifespan(app):
        async with mcp.session_manager.run():
            yield
        studio.close()

    app = FastAPI(title="OzzyImage Lab", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.studio, app.state.mcp = studio, mcp

    @app.middleware("http")
    async def auth(request: Request, call_next):
        path = request.url.path
        protected = path.startswith(("/api/", "/mcp"))
        bearer = request.headers.get("authorization", "")
        bearer_ok = hmac.compare_digest(bearer, "Bearer " + token)
        cookie_ok = hmac.compare_digest(request.cookies.get("ozzy_session", ""), cookie_key)
        if protected and not (bearer_ok or cookie_ok):
            return JSONResponse({"detail": "Oturum açın."}, status_code=401)
        if request.method not in {"GET", "HEAD", "OPTIONS"} and not bearer_ok:
            origin = request.headers.get("origin")
            if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Origin reddedildi."}, status_code=403)
            if protected and request.headers.get("x-ozzy-client") != "studio":
                return JSONResponse({"detail": "İstek doğrulanamadı."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        if protected:
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(FileNotFoundError)
    async def missing(request, exc):
        return JSONResponse({"detail": "Dosya bulunamadı."}, status_code=404)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    class Login(BaseModel):
        token: str = Field(max_length=512)

    @app.post("/auth/login")
    def login(body: Login, request: Request):
        if not hmac.compare_digest(body.token, token):
            return JSONResponse({"detail": "Anahtar hatalı."}, status_code=401)
        response = JSONResponse({"ok": True})
        response.set_cookie("ozzy_session", cookie_key, httponly=True, samesite="strict",
                            secure=request.url.scheme == "https", max_age=43200)
        return response

    @app.post("/auth/logout")
    def logout():
        response = JSONResponse({"ok": True})
        response.delete_cookie("ozzy_session")
        return response

    @app.get("/api/capabilities")
    def api_capabilities():
        return capabilities()

    @app.post("/api/assets")
    async def upload(request: Request):
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > 20 * 1024 * 1024:
                raise ValueError("Dosya en fazla 20 MB olabilir.")
            chunks.append(chunk)
        return studio.store.upload(b"".join(chunks))

    @app.get("/api/assets/{ident}")
    def asset(ident: str):
        return FileResponse(studio.store.path("assets", ident))

    @app.get("/api/outputs/{ident}")
    def output(ident: str):
        return FileResponse(studio.store.path("outputs", ident), media_type="image/png")

    @app.get("/api/outputs/{ident}/recipe")
    def recipe(ident: str):
        return get_recipe(ident)

    @app.post("/api/outputs/{ident}/reuse")
    def reuse(ident: str):
        return reuse_output(ident)

    @app.post("/api/jobs", status_code=202)
    def submit(body: Generation):
        return studio.submit(body)

    @app.get("/api/jobs")
    def jobs():
        return studio.list_jobs()

    @app.get("/api/jobs/{ident}")
    def job(ident: str):
        return studio.get(ident)

    @app.post("/api/jobs/{ident}/cancel")
    def cancel(ident: str):
        return studio.cancel(ident)

    class Preset(BaseModel):
        name: str
        settings: Generation

    @app.get("/api/presets")
    def presets():
        return list_presets()

    @app.post("/api/presets")
    def preset(body: Preset):
        return save_preset(body.name, body.settings)

    @app.delete("/api/presets/{ident}")
    def remove_preset(ident: str):
        return delete_preset(ident)

    app.mount("/mcp", mcp.streamable_http_app())
    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="studio")
    return app
