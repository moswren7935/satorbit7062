from pathlib import Path
import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--cpu-only", action="store_true", help="run CPU tests without probing/building CUDA")


def pytest_configure(config):
    if config.getoption("--cpu-only"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from cubcr import engine_gpu
        engine_gpu.device_count = lambda: 0


def pytest_collection_modifyitems(config, items):
    root = Path(__file__).resolve().parent
    for item in items:
        if (item.path.name == "test_e3b.py" and item.name == "test_active_bench_shape"
                and not (root / "datasets/prism_csg/manifest.json").is_file()):
            item.add_marker(pytest.mark.skip(reason="optional PRISM benchmark files: run python datasets/fetch.py"))
