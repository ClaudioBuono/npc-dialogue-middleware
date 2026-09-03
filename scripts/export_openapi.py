import os
import sys
import json
from pathlib import Path

# Add src to sys.path so we can import from it
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import app

def export_openapi():
    # Get the openapi schema
    openapi_schema = app.openapi()
    
    # Create dist directory if it doesn't exist
    dist_dir = Path(__file__).parent.parent / "dist"
    dist_dir.mkdir(exist_ok=True)
    
    # Write to dist/openapi.json
    output_path = dist_dir / "openapi.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2)
        
    print(f"OpenAPI documentation exported to {output_path}")

if __name__ == "__main__":
    export_openapi()
