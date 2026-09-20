"""FastAPI application for skin scan analysis."""
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .deps import get_settings, setup_logging, CORS_ORIGINS
from .schemas import ScanResponse, HealthResponse
from .utils_io import read_image_bgr
from ..pipeline.compose import run_scan

# Setup
settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

# Create app
app = FastAPI(
    title="Skin Scan OSS",
    description="Production-grade skin analysis API",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    from ..ml.glowlytics import get_engine

    engine = get_engine()
    return {"ok": True, "ml_loaded": engine.available}


@app.post("/scan", response_model=ScanResponse)
async def scan(image: UploadFile = File(...)):
    """
    Analyze uploaded facial image and return skin analysis results.

    Returns:
    - scores: Dict of category scores [0, 1]
    - overlays: Dict of base64-encoded PNG heatmap overlays
    - regions: List of detected facial regions
    """
    try:
        # Read image
        image_data = await image.read()
        logger.info(f"Received image: {image.filename}, size: {len(image_data)} bytes")

        # Check size
        if len(image_data) > settings.max_image_size * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large. Max size: {settings.max_image_size}MB"
            )

        # Convert to BGR
        img_bgr = read_image_bgr(image_data)
        logger.info(f"Image shape: {img_bgr.shape}")

        # Run scan pipeline
        result = run_scan(img_bgr)
        logger.info(f"Scan complete. Scores: {result['scores']}")

        return result

    except ValueError as e:
        logger.error(f"Scan error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception(f"Unexpected error during scan: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during scan")


@app.get("/")
async def root():
    """Serve the demo web UI."""
    web_dir = Path(__file__).parent.parent.parent / "web"
    index_file = web_dir / "client.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Skin Scan API is running. Visit /docs for API documentation."}


# Mount static files for web UI
try:
    web_dir = Path(__file__).parent.parent.parent / "web"
    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")
except Exception as e:
    logger.warning(f"Could not mount web directory: {e}")


@app.on_event("startup")
async def startup_event():
    """Log startup."""
    logger.info(f"Starting Skin Scan API in {settings.env} mode")


@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown."""
    logger.info("Shutting down Skin Scan API")
