from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, field_validator
from app.db.models import UserRole
import re

class Token(BaseModel):
    access_token: str
    token_type: str
    role: UserRole

class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[UserRole] = None  # Ajout du rôle dans les données du token

class UserBase(BaseModel):
    email: EmailStr
    first_name: str = Field(..., min_length=2, example="John")
    last_name: str = Field(..., min_length=2, example="Doe")

class UserCreate(UserBase):
    password: str = Field(
        ...,
        min_length=8,
        description="Doit contenir au moins 1 majuscule, 1 minuscule, 1 chiffre et 1 caractère spécial"
    )
    admin_secret: Optional[str] = Field(  # <-- Ajouter ce champ
        None,
        description="Requis pour créer un compte administrateur"
    )
    company: Optional[str] = None
    phone_number: Optional[str] = None
    
    @field_validator('password')
    def validate_password_complexity(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError("Au moins une majuscule requise")
        if not any(c.islower() for c in v):
            raise ValueError("Au moins une minuscule requise")
        if not any(c.isdigit() for c in v):
            raise ValueError("Au moins un chiffre requis")
        if not any(c in "@$!%*?&" for c in v):
            raise ValueError("Au moins un caractère spécial (@$!%*?&) requis")
        return v

class UserResponse(UserBase):
    id: int
    role: UserRole
    company: Optional[str]
    phone_number: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True
        populate_by_name = True

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., description="User password")

# Ajout pour la réinitialisation de mot de passe
class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)

    @field_validator('new_password')
    def validate_new_password(cls, v):
        # Réutiliser la même validation que pour le mot de passe
        return UserCreate.validate_password_complexity(v)