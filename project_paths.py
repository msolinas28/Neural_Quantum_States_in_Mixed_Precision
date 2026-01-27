from pathlib import Path

def find_project_root(target="mixed_precision_nqs", start_path=None):
    """
    Returns the  '../' needed to reach the folder named `target`
    starting from `start_path` (default: current working directory).
    """
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path).resolve()

    count = ""
    current = start_path

    while current.name != target:
        if current.parent == current:
            raise FileNotFoundError(f"Could not find a folder named '{target}' in any parent directories.")
        current = current.parent
        count += "../"

    return count

PROJECT_ROOT = find_project_root()