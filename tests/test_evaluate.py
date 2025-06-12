import os
import sys
import importlib.util
import pytest

EVAL_PATH = os.path.join(os.path.dirname(__file__), "..", "athenakit", "utils", "evaluate.py")
spec = importlib.util.spec_from_file_location("evaluate", EVAL_PATH)
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)
eval_expr = evaluate.eval_expr


def test_eval_simple_expression():
    vars = {"x": 5, "y": 9}
    result = eval_expr("2 * x + y / 3", lambda v: vars[v])
    assert result == 2 * 5 + 9 / 3


def test_eval_unary_and_mod():
    vars = {"a": 7, "b": 3}
    result = eval_expr("-a % b", lambda v: vars[v])
    assert result == (-7) % 3
