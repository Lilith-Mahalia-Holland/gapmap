from pathlib import Path
from pydantic import DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict

def find_project_root(current_path: Path) -> Path:
    for parent in current_path.parents:
        if (parent / "pyproject.toml").exists() or (parent / "src").exists():
            return parent
    return Path.cwd()

PROJECT_ROOT = find_project_root(Path(__file__).resolve())

class StorageSettings(BaseSettings):
    STORAGE_PATH: DirectoryPath

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def ingest_dir(self) -> Path:
        return self.STORAGE_PATH / "ingest"

    @property
    def models_dir(self) -> Path:
        return self.STORAGE_PATH / "models"

    @property
    def processed_dir(self) -> Path:
        return self.STORAGE_PATH / "processed"


settings = StorageSettings()