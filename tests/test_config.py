from __future__ import annotations

from pathlib import Path

import pytest

from doodl.config import Settings


def test_defaults_are_valid() -> None:
    settings = Settings()
    assert settings.backend == "auto"
    assert settings.max_upload_bytes == 200 * 1024 * 1024
    assert 1 <= settings.max_objects <= 8


def test_from_env_parses_each_type() -> None:
    env = {
        "DOODL_BACKEND": "classical",
        "DOODL_MAX_UPLOAD_MB": "50",
        "DOODL_MAX_DURATION_S": "12.5",
        "DOODL_DATA_DIR": "/tmp/doodl-test",
        "DOODL_ACCESS_TOKEN": "abc",
        "DOODL_SAM2_MODEL": "small",
        "UNRELATED": "ignored",
    }
    settings = Settings.from_env(env)
    assert settings.backend == "classical"
    assert settings.max_upload_mb == 50
    assert settings.max_duration_s == 12.5
    assert settings.data_dir == Path("/tmp/doodl-test")
    assert settings.access_token == "abc"
    assert settings.sam2_model == "small"


def test_empty_env_values_fall_back_to_defaults() -> None:
    assert Settings.from_env({"DOODL_BACKEND": ""}).backend == "auto"


def test_invalid_number_names_the_variable() -> None:
    with pytest.raises(ValueError, match="DOODL_MAX_FRAMES"):
        Settings.from_env({"DOODL_MAX_FRAMES": "lots"})


@pytest.mark.parametrize(
    "changes",
    [
        {"backend": "gpt"},
        {"sam2_model": "huge"},
        {"precision": "fp8"},
        {"prompt_strategy": "magic"},
        {"max_objects": 0},
        {"max_objects": 9},
        {"max_duration_s": 0},
    ],
)
def test_invalid_settings_are_rejected(changes: dict) -> None:
    with pytest.raises(ValueError):
        Settings(**changes)


def test_replace_returns_a_new_validated_copy() -> None:
    base = Settings()
    changed = base.replace(max_objects=3)
    assert changed.max_objects == 3
    assert base.max_objects != 3
    with pytest.raises(ValueError):
        base.replace(backend="nope")
