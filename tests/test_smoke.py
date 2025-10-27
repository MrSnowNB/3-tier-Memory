"""
Smoke tests to verify basic infrastructure setup.
Run these after fresh environment setup to confirm everything works.
"""

import sys
import importlib
from pathlib import Path
import pytest

def test_python_version():
    """Test Python version meets requirements."""
    assert sys.version_info >= (3, 12), f"Python 3.12+ required, got {sys.version}"

def test_package_imports():
    """Test that configured packages can be imported."""
    packages = [
        "torch",
        "fastapi",
        "uvicorn",
        "ollama",
        "numpy",
        "pydantic",
        "pytest",
        "psutil",
    ]

    failed_imports = []
    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError as e:
            failed_imports.append(f"{package}: {e}")

    if failed_imports:
        pytest.fail(f"Failed to import packages: {failed_imports}")

def test_project_structure():
    """Test that required directories exist."""
    required_dirs = [
        "src",
        "tests",
        "scripts",
        "logs",
        "templates",
        "docs"
    ]

    missing_dirs = []
    for dir_name in required_dirs:
        if not Path(dir_name).is_dir():
            missing_dirs.append(dir_name)

    if missing_dirs:
        pytest.fail(f"Missing required directories: {missing_dirs}")

def test_source_package():
    """Test that src is importable as a package."""
    try:
        import src
        assert hasattr(src, "__file__")
    except ImportError as e:
        pytest.fail(f"Cannot import src package: {e}")

def test_env_file():
    """Test that .env.example exists and is readable."""
    env_example = Path(".env.example")
    assert env_example.exists(), ".env.example file missing"

    content = env_example.read_text()
    assert "MPS_MEMORY_FRACTION" in content, "MPS_MEMORY_FRACTION not in .env.example"
    assert "OLLAMA_HOST" in content, "OLLAMA_HOST not in .env.example"

def test_living_docs():
    """Test that living documentation exists."""
    docs = [
        "POLICY.md",
        "TROUBLESHOOTING.md",
        "REPLICATION-NOTES.md",
        "TEST-MATRIX.md",
        "README.md",
        "templates/ISSUE-template.md"
    ]

    missing_docs = []
    for doc in docs:
        if not Path(doc).exists():
            missing_docs.append(doc)

    if missing_docs:
        pytest.fail(f"Missing living documentation: {missing_docs}")

def test_tool_config_files():
    """Test that tool configuration files exist."""
    configs = [
        "pyproject.toml",
        ".env.example"
    ]

    missing_configs = []
    for config in configs:
        if not Path(config).exists():
            missing_configs.append(config)

    if missing_configs:
        pytest.fail(f"Missing configuration files: {missing_configs}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
