# Script to check database schema
import asyncio
import os
import sys

# Add the parent directory to sys.path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.database import AsyncSessionLocal
from sqlalchemy import text

async def check_table_schema(table_name):
    """Check the schema of a specific table"""
    async with AsyncSessionLocal() as session:
        # Query to get column information
        query = text(f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
        """)
        
        result = await session.execute(query)
        columns = result.fetchall()
        
        print(f"\nColumns in {table_name}:")
        for col in columns:
            print(f"- {col[0]} ({col[1]})")
        
        return [col[0] for col in columns]

async def main():
    """Main function to check schemas"""
    print("Checking database schema...")
    
    # Check outlier_detection_runs table
    outlier_columns = await check_table_schema("outlier_detection_runs")
    
    print("\nCode to create OutlierDetectionRun with only existing columns:")
    print("run = OutlierDetectionRun(")
    print("    dataset_id=dataset.id,")
    print("    user_id=current_user.id,")
    print("    status=\"pending\"")
    print(")")
    
    if "parameters" in outlier_columns:
        print("\n# Set parameters separately")
        print("run.parameters = {")
        print("    \"outlier_detection_method\": algorithm,")
        print("    \"parameters\": parsed_parameters,")
        print("    \"save_visualizations\": save_visualizations,")
        print("    \"include_visualizations\": include_visualizations")
        print("}")
    
    # Check feature_engineering table
    print("\nChecking feature_sets table...")
    await check_table_schema("feature_sets")

if __name__ == "__main__":
    asyncio.run(main())
