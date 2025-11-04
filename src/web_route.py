from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path

router = APIRouter()

@router.get("/web", response_class=HTMLResponse)
async def web_root(request: Request):
    html_path = Path(__file__).parent / "web" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Testing-S2S</h1><p>UI not found.</p>")

# Serve static assets under /web/
@router.get("/web/{asset}")
async def web_asset(asset: str):
    asset_path = Path(__file__).parent / "web" / asset
    if asset_path.exists():
        return FileResponse(asset_path)
    return HTMLResponse(status_code=404, content=f"Asset not found: {asset}")
