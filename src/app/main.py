"""FastAPI application for skin scan analysis."""
import logging
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, Header, UploadFile, File, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

from .deps import get_settings, setup_logging, CORS_ORIGINS
from .schemas import ScanResponse, HealthResponse
from .utils_io import read_image_bgr
from ..pipeline.compose import run_scan

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

API_DESCRIPTION = """
## Skin Scan API

Upload a face image with header `X-API-Key` and receive full analysis:

- **scores** — every metric 0–1
- **issues** — prioritized concerns + severity
- **concern_tags** — match these to your cream/product ingredients
- **overlays** — base64 PNG heatmaps
- **detections** — acne lesion boxes
- **profile** — skin type + top issues

### Auth
Set `API_KEY` in server env. Click **Authorize** in Swagger and paste the key, or send:

`X-API-Key: your-secret`

### Visual result viewer
Open [`/playground`](/playground) to upload an image and see heatmaps + concerns rendered.

### curl
```bash
curl -X POST "$HOST/scan" \\
  -H "X-API-Key: YOUR_KEY" \\
  -F "image=@face.jpg"
```
"""

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

app = FastAPI(
    title="Skin Scan API",
    description=API_DESCRIPTION,
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    swagger_ui_parameters={"persistAuthorization": True},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(
    x_api_key: str | None = Security(api_key_header),
    authorization: str | None = Header(default=None),
) -> None:
    expected = (settings.api_key or "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="API_KEY is not configured on the server",
        )
    provided = (x_api_key or "").strip()
    if not provided and authorization:
        auth = authorization.strip()
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
        else:
            provided = auth
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Public health check (no API key)."""
    from ..ml.glowlytics import get_engine
    from ..ml.vit_panel import get_vit_panel

    engine = get_engine()
    panel = get_vit_panel()
    models = []
    if engine.available:
        models.append("glowlytics")
    if panel.available:
        models.append("vit_panel")
    return {
        "ok": True,
        "ml_loaded": engine.available,
        "vit_loaded": panel.available,
        "models": models,
    }


@app.post(
    "/scan",
    response_model=ScanResponse,
    tags=["scan"],
    summary="Full skin scan",
    response_description="Scores, issues, concern tags, heatmaps, detections",
)
async def scan(
    image: UploadFile = File(..., description="Face image (jpg/png/webp)"),
    _: None = Depends(require_api_key),
):
    """
    Run the full multi-model skin pipeline.

    Returns heatmaps in `overlays` as `data:image/png;base64,...` plus
    `issues` / `concern_tags` for product recommendation.
    """
    try:
        image_data = await image.read()
        logger.info(f"Received image: {image.filename}, size: {len(image_data)} bytes")

        if len(image_data) > settings.max_image_size * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large. Max size: {settings.max_image_size}MB",
            )

        img_bgr = read_image_bgr(image_data)
        logger.info(f"Image shape: {img_bgr.shape}")

        result = run_scan(img_bgr)
        logger.info(f"Scan complete. Scores: {result['scores']}")
        return result

    except ValueError as e:
        logger.error(f"Scan error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Unexpected error during scan: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during scan")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs", status_code=307)


@app.get("/playground", include_in_schema=False)
async def playground():
    web_dir = Path(__file__).parent.parent.parent / "web"
    index_file = web_dir / "docs.html"
    if index_file.exists():
        return FileResponse(index_file)
    return RedirectResponse(url="/docs", status_code=307)


try:
    web_dir = Path(__file__).parent.parent.parent / "web"
    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")
except Exception as e:
    logger.warning(f"Could not mount web directory: {e}")


@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting Skin Scan API in {settings.env} mode")
    if not (settings.api_key or "").strip():
        logger.warning("API_KEY is empty — /scan will reject all requests")
    from ..ml.glowlytics import get_engine
    from ..ml.vit_panel import get_vit_panel

    engine = get_engine()
    panel = get_vit_panel()
    logger.info("Models ready glowlytics=%s vit=%s", engine.available, panel.available)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Skin Scan API")
