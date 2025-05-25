# File: backend/app/dependencies.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select  # <-- ADD THIS LINE
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from app.db.database import AsyncSessionLocal, get_db
from app.db.models import User
from app.config import get_settings

settings = get_settings()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Log token details for debugging
        print(f"[DEBUG] Token received: {token[:10]}...")
        print(f"[DEBUG] SECRET_KEY: {settings.SECRET_KEY[:5]}...")
        print(f"[DEBUG] ALGORITHM: {settings.ALGORITHM}")
        print(f"[DEBUG] Full token payload for debugging: {token}")
        
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        print(f"[DEBUG] Decoded payload: {payload}")
        
        # The token contains user_id as subject (not email)
        user_id_str: str = payload.get("sub")
        if user_id_str is None:
            print(f"[DEBUG] User ID not found in token payload")
            raise credentials_exception
            
        try:
            user_id = int(user_id_str)
            print(f"[DEBUG] Successfully decoded token for user_id: {user_id}")
        except ValueError:
            print(f"[DEBUG] Could not convert user_id to int: {user_id_str}")
            raise credentials_exception
            
    except JWTError as e:
        print(f"[DEBUG] JWT Error: {str(e)}")
        raise credentials_exception

    # Fetch user by ID (not email)
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise credentials_exception
    return user