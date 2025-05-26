import asyncio
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import AsyncSessionLocal
from app.db.models import User, UserRole
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def create_user():
    async with AsyncSessionLocal() as session:
        async with session.begin():
            result = await session.execute(
                select(User).where(User.email == "user@gmail.com")
            )
            existing_user = result.scalar_one_or_none()

            if existing_user:
                print("❌ Utilisateur déjà existant : user@gmail.com")
                return

            user = User(
                email="user@gmail.com",
                first_name="user",
                last_name="user1",
                hashed_password=pwd_context.hash("user1234"),
                role=UserRole.USER,
                company="ensa tanger",
                phone_number="0627913833",
                is_active=True,
                created_at=datetime.utcnow()
            )

            session.add(user)
            await session.commit()
            print("✅ Utilisateur créé avec succès : user@gmail.com")

if __name__ == "__main__":
    asyncio.run(create_user())
