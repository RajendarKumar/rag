from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def load_environment() -> None:
    """Load an environment file from the working directory or project root."""
    environment_file = find_dotenv(usecwd=True)
    if not environment_file:
        environment_file = str(Path(__file__).resolve().parents[2] / ".env")
    load_dotenv(environment_file)
