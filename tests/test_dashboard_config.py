from pathlib import Path

import pytest

import dashboard


def test_serene_api_config_from_token_creates_cache_folder(tmp_path):
    cache_folder = tmp_path / "serene-cache"

    config = dashboard._serene_api_config_from_token("secret-token", cache_folder)

    assert config["api"]["token"] == "secret-token"
    assert config["api"]["timeout"] == "30"
    assert config["cache"]["folder"] == str(cache_folder)
    assert config["cache"]["subfolder"] == "{yyyy}/{doy}/"
    assert Path(config["cache"]["folder"]).exists()


def test_data_source_options_put_serene_first_and_hide_missing_local_data(tmp_path):
    assert dashboard._data_source_options(tmp_path) == ["SERENE Latest Ultra-Rapid", "SERENE Latest Rapid"]


def test_data_source_options_include_local_demo_when_h5_files_exist(tmp_path):
    (tmp_path / "sample.h5").write_bytes(b"")

    assert dashboard._data_source_options(tmp_path) == [
        "SERENE Latest Ultra-Rapid",
        "SERENE Latest Rapid",
        "Local demo data",
    ]


def test_missing_serene_config_message_names_streamlit_secret():
    assert "SERENE_API_TOKEN" in dashboard._missing_serene_config_message()


def test_serene_api_config_or_raise_allows_local_default_config(monkeypatch, tmp_path):
    local_config = tmp_path / "api_config.ini"
    local_config.write_text("[api]\ntoken; local-token\n", encoding="utf-8")
    monkeypatch.setattr(dashboard, "default_api_config", lambda: local_config)
    monkeypatch.setattr(dashboard, "_secret_value", lambda name: None)
    monkeypatch.delenv("SERENE_API_TOKEN", raising=False)

    assert dashboard._serene_api_config_or_raise() is None


def test_serene_api_config_or_raise_explains_missing_config(monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard, "default_api_config", lambda: tmp_path / "missing.ini")
    monkeypatch.setattr(dashboard, "_secret_value", lambda name: None)
    monkeypatch.delenv("SERENE_API_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="SERENE_API_TOKEN"):
        dashboard._serene_api_config_or_raise()
