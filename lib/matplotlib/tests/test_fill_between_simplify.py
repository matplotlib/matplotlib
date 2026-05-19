from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest


@dataclass(frozen=True)
class FillScenario:
    name: str
    y1: np.ndarray
    y2: np.ndarray | float
    where: np.ndarray | None = None
    interpolate: bool = False


def _make_scenarios(n=2000, seed=4242):
    t = np.linspace(0, 10, n)

    f1 = np.sin(t)
    f2 = 0.2 * np.cos(2 * t)

    rng = np.random.default_rng(seed)
    where_random = rng.random(n) > 0.3

    c1 = np.sin(t)
    c2 = 0.9 * np.cos(t)

    scenarios = {
        "where_random": FillScenario(
            name="where_random",
            y1=f1,
            y2=f2,
            where=where_random,
        ),
        "interpolate_cross": FillScenario(
            name="interpolate_cross",
            y1=c1,
            y2=c2,
            where=c1 > c2,
            interpolate=True,
        ),
    }

    f3 = np.sin(t) + 0.5
    f4 = -0.5 * np.ones_like(t)

    where_even = np.zeros_like(t, dtype=bool)
    where_even[::2] = True

    where_blocks = np.zeros_like(t, dtype=bool)
    where_blocks[n // 10: 2 * n // 10] = True
    where_blocks[5 * n // 10: 7 * n // 10] = True

    multi_inputs = {
        "t": t,
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "f4": f4,
        "where_even": where_even,
        "where_blocks": where_blocks,
        "where_random": where_random,
    }

    return t, scenarios, multi_inputs


def _save_single_fill(path, t, scenario, threshold):
    with mpl.rc_context({
        "path.simplify": True,
        "path.simplify_threshold": threshold,
    }):
        fig, ax = plt.subplots()
        ax.fill_between(
            t,
            scenario.y1,
            scenario.y2,
            where=scenario.where,
            interpolate=scenario.interpolate,
            alpha=0.5,
        )
        ax.set_xlim(t[0], t[-1])
        fig.savefig(path)
        plt.close(fig)


def _save_multi_fill(path, multi_inputs, threshold):
    t = multi_inputs["t"]

    with mpl.rc_context({
        "path.simplify": True,
        "path.simplify_threshold": threshold,
    }):
        fig, ax = plt.subplots()
        ax.fill_between(
            t, multi_inputs["f1"], multi_inputs["f2"],
            where=multi_inputs["where_blocks"], alpha=0.5,
        )
        ax.fill_between(
            t, multi_inputs["f3"], multi_inputs["f4"],
            where=multi_inputs["where_random"], alpha=0.5,
        )
        ax.fill_between(
            t, multi_inputs["f2"], multi_inputs["f4"],
            where=multi_inputs["where_even"], alpha=0.5,
        )
        ax.set_xlim(t[0], t[-1])
        fig.savefig(path)
        plt.close(fig)


def _assert_smaller(size0, size1, label):
    assert size1 < size0, (
        f"{label}: expected threshold=1.0 output to be smaller than "
        f"threshold=0.0, got {size0} -> {size1}"
    )


@pytest.mark.parametrize("ext", ["svg", "pdf"])
@pytest.mark.parametrize("scenario_key", ["where_random", "interpolate_cross"])
def test_fill_between_simplify_reduces_output_size(tmp_path, ext, scenario_key):
    t, scenarios, _ = _make_scenarios()
    scenario = scenarios[scenario_key]

    path0 = tmp_path / f"{scenario.name}_thr0.{ext}"
    path1 = tmp_path / f"{scenario.name}_thr1.{ext}"

    _save_single_fill(path0, t, scenario, threshold=0.0)
    _save_single_fill(path1, t, scenario, threshold=1.0)

    _assert_smaller(
        path0.stat().st_size,
        path1.stat().st_size,
        f"{scenario.name} {ext}",
    )


@pytest.mark.parametrize("ext", ["svg", "pdf"])
def test_fill_between_multi_regions_simplify_reduces_output_size(tmp_path, ext):
    _, _, multi_inputs = _make_scenarios()

    path0 = tmp_path / f"multi_regions_thr0.{ext}"
    path1 = tmp_path / f"multi_regions_thr1.{ext}"

    _save_multi_fill(path0, multi_inputs, threshold=0.0)
    _save_multi_fill(path1, multi_inputs, threshold=1.0)

    _assert_smaller(
        path0.stat().st_size,
        path1.stat().st_size,
        f"multi_regions {ext}",
    )
