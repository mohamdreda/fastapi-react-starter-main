from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..db.database import get_db
from ..services.auth import (
    authenticate_user,
    create_access_token,
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

    return {
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

@router.post("/register", response_model=dict)
async def register(
    email: str,
    password: str,
    first_name: str,
    last_name: str,
    admin_secret: str = None,
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
        "role": current_user.role
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

@router.options("/login")
async def options_login():
    return {"status": "ok"}