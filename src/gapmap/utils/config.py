from __future__ import annotations


from pathlib import Path
from pydantic import DirectoryPath, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch


def find_project_root(current_path: Path) -> Path:
    for parent in current_path.parents:
        if (parent / "pyproject.toml").exists() or (parent / "src").exists():
            return parent
    return Path.cwd()

PROJECT_ROOT = find_project_root(Path(__file__).resolve())

class ToolSettings(BaseSettings):
    STORAGE_PATH: DirectoryPath
    TRY_DEVICE: str = "cpu"

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

    @computed_field
    @property
    def device(self) -> str:
        if self.TRY_DEVICE != "cpu" and torch.cuda.is_available():
            return "cuda"

        return "cpu"


settings = ToolSettings()