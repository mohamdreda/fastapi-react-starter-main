from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Form, Cookie
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt, JWTError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..db.database import get_db
from ..services.auth import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    get_current_user,
    get_password_hash
)
from ..db.models import User
from ..config import get_settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()
router = APIRouter(tags=["auth"])

@router.post("/login")
async def login(
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Login user and return access token"""
    user = await authenticate_user(email, password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token({"sub": str(user.id)})

    response_data = {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "role": user.role
        }
    }

    # Return JSON response and set HttpOnly refresh token cookie
    resp = JSONResponse(content=response_data)
    resp.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=getattr(settings, "COOKIE_SECURE", False),
        samesite="lax",
        max_age=7 * 24 * 3600,
    )
    return resp

@router.post("/register", response_model=dict)
async def register(
    email: str = Form(...),
    password: str = Form(...),
    first_name: str = Form(...),
    last_name: str = Form(...),
    admin_secret: str = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """Register a new user"""
    # Check if user already exists
    query = select(User).where(User.email == email)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create new user
    user = User(
        email=email,
        first_name=first_name,
        last_name=last_name,
        hashed_password=get_password_hash(password),
        role="admin" if admin_secret == settings.ADMIN_SECRET else "user"
    )

    db.add(user)
    await db.commit()
    await db.refresh(user)

    return {
        "message": "User created successfully",
        "user": {
            "id": user.id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "role": user.role
        }
    }

@router.get("/me", response_model=dict)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "first_name": current_user.first_name,
        "last_name": current_user.last_name,
        "role": current_user.role,
        "company": current_user.company,
        "phone_number": current_user.phone_number,
    }

@router.patch("/me", response_model=dict)
async def update_me(
    payload: dict,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update current user's profile fields (limited to safe attributes)."""
    allowed_fields = {"first_name", "last_name", "company", "phone_number"}
    if payload:
        for field, value in payload.items():
            if field in allowed_fields and value is not None:
                setattr(current_user, field, value)
    await db.commit()
    await db.refresh(current_user)
    return {
        "id": current_user.id,
        "email": current_user.email,
        "first_name": current_user.first_name,
        "last_name": current_user.last_name,
        "role": current_user.role,
        "company": current_user.company,
        "phone_number": current_user.phone_number,
    }

@router.post("/request-password-reset")
async def request_password_reset(email: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).filter(User.email == email))
    user = result.scalar_one_or_none()
    if user:
        user.generate_reset_token()
        await db.commit()
        return {
            "message": "If an account exists with this email, a password reset link will be sent"
        }
    return {"message": "If an account exists with this email, a password reset link will be sent"}

@router.post("/reset-password")
async def reset_password(token: str, new_password: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).filter(User.reset_token == token))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )

    user.set_password(new_password)
    user.clear_reset_token()
    await db.commit()

    return {"message": "Password has been reset successfully"}

@router.post("/change-password")
async def change_password(current_user: User = Depends(get_current_user), old_password: str = Form(...), new_password: str = Form(...), db: AsyncSession = Depends(get_db)):
    if not current_user.verify_password(old_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Old password is incorrect"
        )
    current_user.set_password(new_password)
    await db.commit()
    return {"message": "Password has been changed successfully"}

@router.post("/refresh")
async def refresh_token_endpoint(refresh_token: str = Cookie(None)):
    """Refresh access token using refresh token"""
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Refresh token missing")
    try:
        payload = jwt.decode(refresh_token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user_id = payload.get("sub")
        new_access = create_access_token({"sub": user_id}, timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
        return {"access_token": new_access, "token_type": "bearer"}
    except JWTError:
        raise HTTPException(status_code=401, detail="Refresh token expired or invalid")


@router.options("/login")
async def options_login():
    return {"status": "ok"}