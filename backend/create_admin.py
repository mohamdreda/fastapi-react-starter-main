# create_admin.py

import asyncio
from app.db.database import get_database_url, AsyncSessionLocal, engine
from app.db.models import User, UserRole
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def create_admin():
    async with AsyncSessionLocal() as session:
        admin_email = "admin@gmail.com"
        existing = await session.execute(
            User.__table__.select().where(User.email == admin_email)
        )
        if existing.scalar():
            print("❌ Admin already exists.")
            return

        admin = User(
            email=admin_email,
            first_name="Admin",
            last_name="System",
            hashed_password=pwd_context.hash("Admin@1234"),
            role=UserRole.ADMIN,
            is_active=True,
            created_at=datetime.utcnow()
        )

        session.add(admin)
        await session.commit()
        print("✅ Admin user created successfully.")

if __name__ == "__main__":
    asyncio.run(create_admin())
