"""GET /api/qto-modules — list all available QTO modules with metadata."""

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from src.analysis.qto.registry import list_modules, QTO_REGISTRY

router = APIRouter(tags=["qto"])


@router.get(
    "/api/qto-modules",
    summary="List QTO modules",
    response_description="All registered QTO takeoff modules with metadata",
)
def get_qto_modules(
    enabled_only: bool = Query(default=False, description="Return only enabled modules"),
) -> JSONResponse:
    """
    Return all QTO (Quantity Takeoff) modules registered in the pipeline,
    with metadata: name, agent_id, trades covered, LLM requirement, and
    whether the module is currently enabled.

    Modules can be disabled at runtime via:
      - XBOQ_DISABLE_QTO=name1,name2   (comma-separated)
      - XBOQ_DISABLE_IMPLIED_ITEMS=1   (legacy flag for the implied-items module)
    """
    modules = list_modules(enabled_only=enabled_only)
    return JSONResponse(content={
        "total": len(QTO_REGISTRY),
        "enabled": sum(1 for m in list_modules() if m["enabled"]),
        "modules": modules,
    })
