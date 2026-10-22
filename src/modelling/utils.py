import pickle
from pathlib import Path
from typing import Any


def pickle_object(obj: Any, path: Path) -> None:
    """Save a Python object to a pickle file, creating the parent folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    """Load a Python object from a pickle file."""
    with path.open("rb") as f:
        return pickle.load(f)
