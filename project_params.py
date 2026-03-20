import json
from pathlib import Path


PARAMS_FILE = Path(__file__).parent / "params" / "projects.json"


def load_project_params(project_name):
    with PARAMS_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    projects = data.get("projects", {})
    if project_name not in projects:
        available = ", ".join(sorted(projects.keys()))
        raise ValueError(f"Unknown project '{project_name}'. Available projects: {available}")

    return projects[project_name]
