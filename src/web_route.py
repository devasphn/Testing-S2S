from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from pathlib import Path

router = APIRouter()

@router.get("/web", response_class=HTMLResponse)
async def web_root(request: Request):
    # Serve embedded minimal UI
    html_path = Path(__file__).parent / "web" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Testing-S2S</h1><p>UI not found.</p>")
