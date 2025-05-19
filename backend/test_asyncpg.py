import asyncio
import asyncpg

async def test():
    try:
        conn = await asyncpg.connect("postgresql://postgres:1234@localhost:5432/cleaning_db")
        print("✅ Connexion réussie à PostgreSQL !")
        await conn.close()
    except Exception as e:
        print("❌ Échec :", e)

asyncio.run(test())
