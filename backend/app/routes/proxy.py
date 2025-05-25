"""
Proxy route to handle CORS issues
This route acts as a proxy for other API endpoints to avoid CORS restrictions
"""

from fastapi import APIRouter, Request, Response, Depends, HTTPException
import httpx
from ..dependencies import get_current_user
from ..config import get_settings
import logging

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

@router.get("/proxy/{path:path}")
async def proxy_request(
    path: str, 
    request: Request,
    current_user = Depends(get_current_user)
):
    """
    Proxy endpoint to forward requests to the actual API
    This helps bypass CORS restrictions
    """
    try:
        # Get the token from the original request
        token = request.headers.get("Authorization")
        if not token:
            raise HTTPException(status_code=401, detail="Authorization header missing")
        
        # Build the target URL
        target_url = f"{settings.API_BASE_URL}/{path}"
        logger.info(f"Proxying request to: {target_url}")
        
        # Forward the request to the target URL
        async with httpx.AsyncClient() as client:
            # Get query parameters from original request
            params = dict(request.query_params)
            
            # Forward the request with the same headers
            response = await client.get(
                target_url,
                headers={"Authorization": token},
                params=params,
                timeout=30.0  # Increased timeout for potentially slow operations
            )
            
            # Return the response from the target URL
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("content-type", "application/json")
            )
    except httpx.RequestError as exc:
        logger.error(f"Error proxying request: {str(exc)}")
        raise HTTPException(status_code=500, detail=f"Error proxying request: {str(exc)}")
    except Exception as e:
        logger.error(f"Unexpected error in proxy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")
