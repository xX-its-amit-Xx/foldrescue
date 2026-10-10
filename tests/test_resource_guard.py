"""Tests for the resource guard.

The guard's job is to refuse rather than OOM, so the tests that matter most
are the refusal tests. "It refused correctly" is the required behaviour from
the specification, and a guard that quietly proceeds is the bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from foldrescue.utils import (
    ResourcePlan,
    ResourceRefusal,
    check_resources,
    iter_chunks,
)


def test_small_allocation_passes_unmodified():
    plan = check_resources(100, 10, dtype=np.float64)
    assert plan.dtype == np.dtype(np.float64)
    assert plan.chunk_rows == 100
    assert plan.n_chunks == 1
    assert not plan.downcast
    assert not plan.chunked
    assert plan.mitigations == ()


def test_n_jobs_is_capped_at_nproc_minus_one():
    """Every extra worker is another copy of the working set."""
    plan = check_resources(10, 10, n_jobs=10_000)
    import psutil

    assert plan.n_jobs == max(1, psutil.cpu_count() - 1)
    assert plan.n_jobs >= 1
    assert any("n_jobs" in m for m in plan.mitigations)


def test_n_jobs_none_resolves_to_cap():
    plan = check_resources(10, 10, n_jobs=None)
    assert plan.n_jobs >= 1
    # Requesting the default is not a mitigation, so nothing is recorded.
    assert not any("n_jobs" in m for m in plan.mitigations)


def test_downcast_is_the_first_mitigation():
    """A float64 request slightly over budget becomes float32, not chunked."""
    avail = check_resources(1, 1).available_bytes
    # Pick a width whose float64 footprint exceeds half of available RAM but
    # whose float32 footprint does not.
    budget = avail * 0.5
    n_cols = int(budget / (8 * 2) * 1.5)  # 1.5x over in float64, 0.75x in float32
    plan = check_resources(1, n_cols, dtype=np.float64)
    assert plan.dtype == np.dtype(np.float32)
    assert plan.downcast
    assert any("downcast" in m for m in plan.mitigations)
    assert plan.n_chunks == 1, "downcast alone should have been sufficient"


def test_chunking_when_downcast_is_not_enough():
    avail = check_resources(1, 1).available_bytes
    # One row fits comfortably in float32; a million of them do not.
    row_cols = 4096
    row_bytes_f32 = row_cols * 4 * 2
    n_rows = int((avail * 0.5) / row_bytes_f32 * 8)  # 8x over budget
    plan = check_resources(n_rows, row_cols, dtype=np.float64)
    assert plan.chunked
    assert plan.chunk_rows < n_rows
    assert plan.n_chunks > 1
    assert plan.estimated_bytes <= plan.budget_bytes


def test_refuses_when_a_single_row_cannot_fit():
    """The required refusal: actionable, and carrying structured fields."""
    avail = check_resources(1, 1).available_bytes
    # One row wider than the entire budget, even in float32.
    n_cols = int(avail)  # float32 x 2 copies => ~8x available
    with pytest.raises(ResourceRefusal) as excinfo:
        check_resources(10, n_cols, dtype=np.float64)

    exc = excinfo.value
    assert exc.required_bytes > exc.budget_bytes
    assert exc.remedies, "a refusal with no remedies is not actionable"
    message = str(exc)
    # The message must name the numbers and at least one concrete knob.
    assert "refuses" in message
    assert "memory_fraction" in message
    assert any(unit in message for unit in ("GiB", "MiB", "TiB"))
    # It is a MemoryError so generic handlers still behave.
    assert isinstance(exc, MemoryError)


def test_refuses_when_chunking_is_disallowed_and_needed():
    avail = check_resources(1, 1).available_bytes
    row_cols = 4096
    n_rows = int((avail * 0.5) / (row_cols * 4 * 2) * 8)
    with pytest.raises(ResourceRefusal) as excinfo:
        check_resources(n_rows, row_cols, allow_chunking=False)
    assert any("chunking" in r for r in excinfo.value.remedies)


def test_no_downcast_flag_is_respected():
    """With downcast disabled the guard chunks instead, keeping float64.

    Note the shape: several rows, each individually affordable. With a single
    row there is nothing to chunk, and refusing is the correct answer -- which
    is what an earlier version of this test wrongly asserted against.
    """
    avail = check_resources(1, 1).available_bytes
    n_cols = 2048
    row_bytes_f64 = n_cols * 8 * 2
    n_rows = int((avail * 0.5) / row_bytes_f64 * 4)  # 4x over budget in total
    plan = check_resources(n_rows, n_cols, dtype=np.float64, allow_downcast=False)

    assert plan.dtype == np.dtype(np.float64), "downcast was disabled"
    assert not plan.downcast
    assert plan.chunked, "the guard should have chunked instead of downcasting"
    assert plan.estimated_bytes <= plan.budget_bytes
    assert not any("downcast" in m for m in plan.mitigations)


def test_memory_fraction_scales_the_budget():
    tight = check_resources(1000, 1000, memory_fraction=0.01)
    loose = check_resources(1000, 1000, memory_fraction=0.9)
    assert tight.budget_bytes < loose.budget_bytes


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_invalid_memory_fraction_rejected(bad):
    with pytest.raises(ValueError, match="memory_fraction"):
        check_resources(10, 10, memory_fraction=bad)


def test_negative_shape_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        check_resources(-1, 10)


def test_degenerate_shapes_do_not_refuse():
    """A zero-row frame allocates nothing and must not trip the refusal path."""
    assert check_resources(0, 100).n_chunks == 0
    assert check_resources(100, 0).estimated_bytes == 0


def test_iter_chunks_covers_every_row_exactly_once():
    plan = ResourcePlan(
        dtype=np.dtype(np.float32),
        chunk_rows=7,
        n_jobs=1,
        n_chunks=3,
        estimated_bytes=0,
        budget_bytes=1,
        available_bytes=1,
        downcast=False,
    )
    seen = np.zeros(20, dtype=int)
    for sl in iter_chunks(20, plan):
        seen[sl] += 1
    assert (seen == 1).all()


def test_working_copies_multiplier_increases_the_estimate():
    one = check_resources(1000, 1000, working_copies=1)
    two = check_resources(1000, 1000, working_copies=2)
    assert two.estimated_bytes == 2 * one.estimated_bytes


def test_plan_describe_is_informative():
    plan = check_resources(10, 10)
    text = plan.describe()
    for token in ("dtype=", "chunk_rows=", "n_jobs=", "est=", "budget="):
        assert token in text
