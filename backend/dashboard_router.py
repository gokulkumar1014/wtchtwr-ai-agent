"""FastAPI router exposing dashboard analytics endpoints on the main HOPE backend."""
from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from .dashboard import DashboardRequest, build_dashboard_response, load_filter_options
logger = logging.getLogger('hope.dashboard')
router = APIRouter()

@router.get('/meta')
async def dashboard_meta() -> JSONResponse:
    """Return filter metadata required to initialise the dashboard UI."""
    try:
        payload = load_filter_options()
        return JSONResponse(content=payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception('Unable to load dashboard filter metadata: %s', exc)
        raise HTTPException(status_code=500, detail='Unable to load dashboard metadata.') from exc

@router.get('/filters')
async def dashboard_filters_alias() -> JSONResponse:
    """Alias for backward compatibility with legacy frontend filter endpoint."""
    logger.info('⚙️  Dashboard filters alias hit — redirecting to /meta')
    return await dashboard_meta()

@router.post('/insights')
async def dashboard_insights(request: DashboardRequest) -> JSONResponse:
    """Return dashboard insights for the provided filter/comparison payload."""
    try:
        response = build_dashboard_response(request)
        return JSONResponse(content=response)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception('Unable to build dashboard insights: %s', exc)
        raise HTTPException(status_code=500, detail='Unable to build dashboard insights.') from exc

@router.post('/view')
async def dashboard_view_alias(request: DashboardRequest) -> JSONResponse:
    """Alias for legacy dashboard view endpoint used by older frontend builds."""
    logger.info('⚙️  Dashboard view alias hit — redirecting to /insights')
    return await dashboard_insights(request)