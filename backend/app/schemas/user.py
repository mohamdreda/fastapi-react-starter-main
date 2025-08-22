# backend/app/schemas/user.py
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime # Import datetime
from app.db.models import UserRole # Assuming UserRole is in db.models (your original import)

class UserBase(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str
    # role: str # Let's use the enum for better type safety if possible
    company: Optional[str] = None # Added from your DB model
    phone_number: Optional[str] = None # Added from your DB model

class UserCreate(UserBase):
    password: str
    role: UserRole = UserRole.USER # Default role on creation

class UserUpdate(BaseModel): # Keep this separate for update flexibility
    email: Optional[EmailStr] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: Optional[UserRole] = None # Use the enum here too
    company: Optional[str] = None
    phone_number: Optional[str] = None
    is_active: Optional[bool] = None
    # Password update should ideally be a separate, more secure endpoint/process

# This will be the Pydantic model for representing a User, e.g., for responses or for `current_user`
class User(UserBase): # << RENAME UserInDB to User, or create a new User schema
    id: int
    is_active: bool       # Added from your DB model
    role: UserRole        # Use the enum
    created_at: datetime  # Added from your DB model
    # company: Optional[str] = None # Already in UserBase if inherited
    # phone_number: Optional[str] = None # Already in UserBase if inherited

    class Config:
        from_attributes = True # Pydantic V2 (replaces orm_mode = True)
        use_enum_values = True # To serialize enum members to their values if needed in responses

# You can keep UserInDB if it serves a different purpose, or rename it.
# For clarity, let's assume User is the primary schema for fetched user data.
# If UserInDB was meant for this, rename UserInDB to User.
# For this example, I'm assuming we define User as above.

# Your UserInDB was:
# class UserInDB(UserBase):
#     id: int
#     # This UserBase had role: str, ensure consistency or update UserBase
#     class Config:
#         orm_mode = True # change to from_attributes = True