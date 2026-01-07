import os
from pathlib import Path

# Define the directory structure
structure = {
    "retail_predictor": {
        "files": ["app.py", "models.py", "requirements.txt"],
        "templates": [
            "base.html",
            "index.html",
            "login.html",
            "register.html",
            "dashboard.html",
            "prediction.html",
            "results.html",
            "analysis.html"
        ],
        "static": {
            "css": ["style.css"],
            "js": ["script.js"],
            "images": []
        },
        "instance": ["retail.db"],
        "models": [
            "random_forest_model.pkl",
            "xgboost_model.pkl",
            "lightgbm_model.pkl",
            "catboost_model.pkl",
            "preprocessing_objects.pkl",
            "model_results.json"
        ]
    }
}

def create_structure(base_path, structure):
    for folder, content in structure.items():
        folder_path = Path(base_path) / folder
        folder_path.mkdir(parents=True, exist_ok=True)

        # Create root-level files
        if "files" in content:
            for file in content["files"]:
                (folder_path / file).touch()

        # Create templates and files
        if "templates" in content:
            templates_path = folder_path / "templates"
            templates_path.mkdir(exist_ok=True)
            for file in content["templates"]:
                (templates_path / file).touch()

        # Create static directories and files
        if "static" in content:
            static_path = folder_path / "static"
            static_path.mkdir(exist_ok=True)
            for subfolder, files in content["static"].items():
                sub_path = static_path / subfolder
                sub_path.mkdir(exist_ok=True)
                for file in files:
                    (sub_path / file).touch()

        # Create instance folder and DB file
        if "instance" in content:
            instance_path = folder_path / "instance"
            instance_path.mkdir(exist_ok=True)
            for file in content["instance"]:
                (instance_path / file).touch()

        # Create models folder and model files
        if "models" in content:
            models_path = folder_path / "models"
            models_path.mkdir(exist_ok=True)
            for file in content["models"]:
                (models_path / file).touch()

# Run the creation
create_structure(".", structure)

print("✅ retail_predictor project directory created successfully!")
