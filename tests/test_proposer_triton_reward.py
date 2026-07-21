import importlib.util
import os
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "examples" / "reward_function" / "proposer_triton.py"
spec = importlib.util.spec_from_file_location("proposer_triton", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_extract_python_last_block():
    text = "thinking\n```python\nclass Model: pass\n```"
    assert module._extract_python(text) == "class Model: pass"


def test_frontier_reward_matches_paper(monkeypatch):
    monkeypatch.setenv("PROPOSER_FRONTIER_TARGET", "0.5")
    assert module._frontier_reward(0.0) == 0.0
    assert module._frontier_reward(0.5) == 1.0
    assert module._frontier_reward(1.0) == 0.0


def test_invalid_module_gets_penalty(monkeypatch):
    monkeypatch.setenv("PROPOSER_INVALID_REWARD", "-1")
    assert module._score_one("not python code") == {
        "overall": -1.0,
        "validity": 0.0,
        "coder_accuracy": 0.0,
    }
