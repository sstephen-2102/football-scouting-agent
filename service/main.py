"""Standalone HTTP service.

Run from the repository root:

    python -m service.main
"""

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from service.routes.capabilities import router as capabilities_router
from service.routes.system import router as system_router

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    application = FastAPI(
        title="Football Scouting Agent",
        version="0.1.0",
        description=(
            "AAI-501 sporting-director service. Domain code lives in src/; "
            "this process only serves artifacts and decisions."
        ),
    )
    application.include_router(system_router, tags=["system"])
    application.include_router(capabilities_router, tags=["capabilities"])
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return application


app = create_app()


def main() -> None:
    uvicorn.run("service.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
