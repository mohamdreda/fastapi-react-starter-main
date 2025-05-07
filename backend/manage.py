import typer
import subprocess
import os
from sqlalchemy import create_engine
from app.config import get_settings
from app.db import engine, Base

app = typer.Typer()
settings = get_settings()


@app.command()
def run():
    """Run the FastAPI application."""
    typer.echo("Starting FastAPI app...")
    subprocess.run(["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])


@app.command()
def makemigrations(message: str = "Auto migration"):
    """Generate a new Alembic migration."""
    typer.echo(f"Generating migration: {message}")
    subprocess.run(["alembic", "revision", "--autogenerate", "-m", message])


@app.command()
def migrate():
    """Apply all pending Alembic migrations."""
    typer.echo("Applying migrations...")
    subprocess.run(["alembic", "upgrade", "head"])


@app.command()
def downgrade(revision: str = "-1"):
    """Rollback the last Alembic migration (default: one step)."""
    typer.echo(f"Rolling back to {revision}...")
    subprocess.run(["alembic", "downgrade", revision])


@app.command()
def db_status():
    """Show current database migration status."""
    typer.echo("Checking Alembic migration status...")
    subprocess.run(["alembic", "current"])


@app.command()
def format():
    """Format Python code with black."""
    typer.echo("Formatting Python code with black...")
    subprocess.run(["black", "."])


@app.command()
def reset_db():
    """Drop and recreate all database tables using Alembic"""
    typer.echo("Resetting database using Alembic...")
    subprocess.run(["alembic", "downgrade", "base"])
    subprocess.run(["alembic", "upgrade", "head"])
    typer.echo("Database reset successfully.")


if __name__ == "__main__":
    app()
