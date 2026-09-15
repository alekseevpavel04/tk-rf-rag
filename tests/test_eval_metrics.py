import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("run_eval", Path(__file__).parents[1] / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)


@pytest.mark.parametrize(
    "ranked,gold,expected",
    [
        (["115", "116", "117"], {"115"}, (1, 1, 1.0)),  # first place
        (["80", "115", "117"], {"115"}, (0, 1, 0.5)),  # second place -> RR = 1/2
        (["1", "2", "3", "4", "115"], {"115"}, (0, 1, 0.2)),  # fifth place
        (["1", "2", "3", "4", "5", "115"], {"115"}, (0, 0, 0.0)),  # sixth place is outside top-5
        (["80", "81"], {"81", "80"}, (1, 1, 1.0)),  # any gold article counts
        ([], {"115"}, (0, 0, 0.0)),
    ],
)
def test_rank_metrics(ranked, gold, expected):
    h1, h5, rr = run_eval.rank_metrics(ranked, gold)
    assert (h1, h5) == expected[:2]
    assert rr == pytest.approx(expected[2])


@pytest.mark.parametrize("raw,expected", [("1", 1), ("0", 0), (" 1\n", 1), ("Оценка: 0", 0), ("да", None), ("", None)])
def test_parse_judge(raw, expected):
    assert run_eval.parse_judge(raw) == expected


def test_fmt_shows_mean_and_counts():
    assert run_eval.fmt([1, 0, 1, 1]) == "0.750 (3/4)"
    assert run_eval.fmt([0.5, 1.0], as_count=False) == "0.750"
    assert run_eval.fmt([]) == "—"
