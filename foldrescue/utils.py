"""Resource guarding, logging, and small shared helpers.

THE RESOURCE GUARD CONTRACT
===========================

``check_resources()`` is the single gate every allocation-heavy code path in
foldrescue passes through *before* it allocates. The contract is deliberately
narrow and deliberately boring, because the failure it prevents is the one that
kills long-running jobs: a dense ``float64`` matrix that silently exceeds
available RAM, gets swapped, and takes the machine down with it.

The guard does four things, in order:

1.  **Read the machine.** Available RAM via ``psutil.virtual_memory().available``
    (not ``total`` -- what matters is what you can actually get) and logical CPU
    count via ``psutil.cpu_count()``.
2.  **Estimate the dense footprint.** ``n_rows * n_cols * itemsize``, plus a
    working-copy multiplier, because almost nothing operates in place: sklearn
    and LightGBM routinely hold the input and a transformed copy at once.
3.  **Mitigate, in a fixed order.** If the estimate exceeds ``threshold``
    (default 50%) of available RAM:
        a. downcast ``float64 -> float32`` (halves the footprint, and is
           harmless here: no feature in foldrescue carries 15 significant
           digits of real information),
        b. chunk along rows so each chunk fits the budget,
        c. cap parallelism at ``nproc - 1``, because every extra worker process
           is another copy of the working set, and leaving one core free keeps
           the box responsive.
4.  **Refuse, actionably.** If even one single-row chunk of ``float32`` cannot
    fit, raise ``ResourceRefusal`` naming the numbers and the specific knobs
    that would make the job fit. A refusal is a *success* of this function: the
    alternative is an OOM kill thirty minutes into a run.

The guard never silently proceeds with a plan that exceeds the budget, and it
never mutates global state. It returns a ``ResourcePlan`` that the caller is
expected to honour -- ``plan.dtype``, ``plan.chunk_rows``, ``plan.n_jobs``.

Why 50% and not 90%? Because the estimate is a lower bound. Python objects,
BLAS scratch buffers, and the interpreter itself all live outside it. Half of
available memory is the headroom that makes the estimate trustworthy enough to
act on.

This same contract is documented in ``CLAUDE.md`` and ``AGENTS.md`` so that
agents editing this repository cannot "helpfully" allocate around it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ResourceRefusal",
    "ResourcePlan",
    "check_resources",
    "iter_chunks",
    "get_logger",
    "MEMORY_FRACTION_DEFAULT",
    "WORKING_COPIES_DEFAULT",
]

# Fraction of *available* (not total) RAM a dense allocation may claim before
# mitigation kicks in. See module docstring for why this is 0.5.
MEMORY_FRACTION_DEFAULT = 0.5

# Most estimators hold the input plus at least one transformed copy. Two is the
# empirical floor for LightGBM (raw + binned) and sklearn transformers.
WORKING_COPIES_DEFAULT = 2

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str = "foldrescue") -> logging.Logger:
    """Return a configured logger, idempotently.

    Resource decisions are logged at INFO because a user who hits a chunked or
    downcast run deserves to know why their numbers might differ from a
    float64 run elsewhere.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("FOLDRESCUE_LOG_LEVEL", "INFO").upper())
    return logger


_log = get_logger(__name__)


class ResourceRefusal(MemoryError):
    """Raised when a request cannot be satisfied within the memory budget.

    Subclasses ``MemoryError`` so that callers with a generic
    ``except MemoryError`` still behave sensibly, but carries structured fields
    so a CLI can render something actionable instead of a traceback.
    """

    def __init__(
        self,
        message: str,
        *,
        required_bytes: int,
        budget_bytes: int,
        available_bytes: int,
        remedies: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.required_bytes = required_bytes
        self.budget_bytes = budget_bytes
        self.available_bytes = available_bytes
        self.remedies = tuple(remedies)


@dataclass(frozen=True)
class ResourcePlan:
    """The guard's verdict: how the caller must allocate to stay in budget.

    Attributes
    ----------
    dtype:
        The dtype the caller must materialise arrays in. May be downcast from
        the requested dtype.
    chunk_rows:
        Maximum rows per chunk. Equals ``n_rows`` when no chunking is needed.
    n_jobs:
        Parallelism the caller must cap at. Never exceeds ``nproc - 1``.
    n_chunks:
        Number of chunks implied by ``chunk_rows``.
    estimated_bytes:
        Dense footprint of a single chunk under ``dtype``, including working
        copies. This is what was compared against the budget.
    budget_bytes:
        ``memory_fraction * available RAM`` at the time of the check.
    available_bytes:
        Available RAM at the time of the check.
    downcast:
        True if ``dtype`` differs from what the caller requested.
    mitigations:
        Human-readable record of what the guard did, for logging and for the
        ``RUN_ENVIRONMENT.md`` provenance trail.
    """

    dtype: np.dtype
    chunk_rows: int
    n_jobs: int
    n_chunks: int
    estimated_bytes: int
    budget_bytes: int
    available_bytes: int
    downcast: bool
    mitigations: tuple[str, ...] = field(default_factory=tuple)

    @property
    def chunked(self) -> bool:
        return self.n_chunks > 1

    def describe(self) -> str:
        """One-line summary suitable for logs and notebook output."""
        parts = [
            f"dtype={np.dtype(self.dtype).name}",
            f"chunk_rows={self.chunk_rows}",
            f"n_chunks={self.n_chunks}",
            f"n_jobs={self.n_jobs}",
            f"est={_human_bytes(self.estimated_bytes)}",
            f"budget={_human_bytes(self.budget_bytes)}",
        ]
        summary = " ".join(parts)
        if self.mitigations:
            summary += " | mitigations: " + ", ".join(self.mitigations)
        return summary


def _human_bytes(n: float) -> str:
    """Format a byte count for humans. Used in every refusal message."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.2f} TiB"  # pragma: no cover - unreachable, loop returns first


def _read_machine() -> tuple[int, int]:
    """Return ``(available_bytes, cpu_count)``.

    psutil is the documented dependency, but the guard must not itself be the
    reason a run dies, so we degrade to stdlib if psutil is unavailable. The
    degraded path is logged loudly because its ``available`` figure is total
    RAM, which is optimistic -- the guard becomes weaker, not absent.
    """
    try:
        import psutil  # noqa: PLC0415 - optional-at-runtime by design

        return int(psutil.virtual_memory().available), int(psutil.cpu_count() or 1)
    except Exception:  # pragma: no cover - exercised only without psutil
        _log.warning(
            "psutil unavailable; falling back to os.sysconf for the memory read. "
            "The guard will use TOTAL rather than AVAILABLE memory, which is "
            "optimistic. Install psutil for an accurate budget."
        )
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            total = 2 * 1024**30  # last-resort pessimistic 2 GiB
        return int(total), int(os.cpu_count() or 1)


def check_resources(
    n_rows: int,
    n_cols: int,
    *,
    dtype: np.dtype | type = np.float64,
    n_jobs: int | None = None,
    memory_fraction: float = MEMORY_FRACTION_DEFAULT,
    working_copies: int = WORKING_COPIES_DEFAULT,
    allow_downcast: bool = True,
    allow_chunking: bool = True,
    label: str = "allocation",
) -> ResourcePlan:
    """Gate a dense allocation against available memory. See module docstring.

    Parameters
    ----------
    n_rows, n_cols:
        Shape of the dense matrix the caller intends to materialise.
    dtype:
        Requested dtype. Downcast to ``float32`` if needed and permitted.
    n_jobs:
        Requested parallelism. ``None`` or ``-1`` means "all cores". Always
        capped at ``nproc - 1``.
    memory_fraction:
        Fraction of *available* RAM the allocation may claim.
    working_copies:
        Multiplier for transient copies held alongside the input.
    allow_downcast, allow_chunking:
        Disable individual mitigations. Disabling both turns the guard into a
        pure pass/refuse check, which is what the tests use.
    label:
        Name of the call site, for log lines and refusal messages.

    Returns
    -------
    ResourcePlan

    Raises
    ------
    ValueError
        On a nonsensical request (negative shape, ``memory_fraction`` outside
        ``(0, 1]``).
    ResourceRefusal
        When no permitted mitigation brings the request within budget.
    """
    if n_rows < 0 or n_cols < 0:
        raise ValueError(f"shape must be non-negative, got ({n_rows}, {n_cols})")
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError(f"memory_fraction must be in (0, 1], got {memory_fraction}")
    if working_copies < 1:
        raise ValueError(f"working_copies must be >= 1, got {working_copies}")

    available, ncpu = _read_machine()
    budget = int(available * memory_fraction)
    mitigations: list[str] = []

    # --- parallelism: always leave one core free -------------------------
    # Every worker is another copy of the working set, so this is a memory
    # decision as much as a CPU one.
    cap = max(1, ncpu - 1)
    if n_jobs is None or n_jobs <= 0:
        resolved_jobs = cap
    else:
        resolved_jobs = min(n_jobs, cap)
    if n_jobs is not None and n_jobs > cap:
        mitigations.append(f"n_jobs {n_jobs}->{resolved_jobs} (nproc-1)")

    # --- footprint estimate ----------------------------------------------
    requested_dtype = np.dtype(dtype)
    row_bytes = int(n_cols) * requested_dtype.itemsize * working_copies
    total_bytes = row_bytes * int(n_rows)

    # Degenerate shapes allocate nothing; short-circuit so a 0-row frame does
    # not trip the refusal path.
    if n_rows == 0 or n_cols == 0:
        return ResourcePlan(
            dtype=requested_dtype,
            chunk_rows=int(n_rows),
            n_jobs=resolved_jobs,
            n_chunks=0 if n_rows == 0 else 1,
            estimated_bytes=0,
            budget_bytes=budget,
            available_bytes=available,
            downcast=False,
            mitigations=tuple(mitigations),
        )

    plan_dtype = requested_dtype
    if total_bytes <= budget:
        plan = ResourcePlan(
            dtype=plan_dtype,
            chunk_rows=int(n_rows),
            n_jobs=resolved_jobs,
            n_chunks=1,
            estimated_bytes=total_bytes,
            budget_bytes=budget,
            available_bytes=available,
            downcast=False,
            mitigations=tuple(mitigations),
        )
        _log.debug("check_resources(%s): within budget | %s", label, plan.describe())
        return plan

    # --- mitigation 1: downcast ------------------------------------------
    if allow_downcast and requested_dtype.kind == "f" and requested_dtype.itemsize > 4:
        plan_dtype = np.dtype(np.float32)
        row_bytes = int(n_cols) * plan_dtype.itemsize * working_copies
        total_bytes = row_bytes * int(n_rows)
        mitigations.append(f"downcast {requested_dtype.name}->float32")

    chunk_rows = int(n_rows)
    if total_bytes > budget:
        # --- mitigation 2: chunk -----------------------------------------
        if not allow_chunking:
            raise _refusal(label, total_bytes, budget, available, row_bytes, chunking_allowed=False)
        # Largest row count whose footprint fits the budget.
        chunk_rows = int(budget // row_bytes) if row_bytes else int(n_rows)
        if chunk_rows < 1:
            # Even one row will not fit: nothing left to try.
            raise _refusal(label, row_bytes, budget, available, row_bytes, chunking_allowed=True)
        mitigations.append(f"chunk rows {n_rows}->{chunk_rows}")
        total_bytes = chunk_rows * row_bytes

    n_chunks = max(1, -(-int(n_rows) // chunk_rows))  # ceil division
    plan = ResourcePlan(
        dtype=plan_dtype,
        chunk_rows=chunk_rows,
        n_jobs=resolved_jobs,
        n_chunks=n_chunks,
        estimated_bytes=total_bytes,
        budget_bytes=budget,
        available_bytes=available,
        downcast=plan_dtype != requested_dtype,
        mitigations=tuple(mitigations),
    )
    _log.info("check_resources(%s): %s", label, plan.describe())
    return plan


def _refusal(
    label: str,
    required: int,
    budget: int,
    available: int,
    row_bytes: int,
    *,
    chunking_allowed: bool,
) -> ResourceRefusal:
    """Build an actionable refusal.

    "Actionable" means the message names the numbers *and* the specific knobs
    that would make the job fit. A refusal the user cannot act on is just an
    OOM with extra steps.
    """
    remedies = [
        f"reduce feature width (one row alone needs {_human_bytes(row_bytes)})",
        "drop a feature block: --no-ligand-block, --no-structural-block",
        "raise the budget with memory_fraction=... if you accept the swap risk",
        "run on a machine with more available RAM",
    ]
    if not chunking_allowed:
        remedies.insert(0, "allow chunking (allow_chunking=True) to stream this in pieces")
    message = (
        f"foldrescue refuses the {label}: it needs {_human_bytes(required)} but the "
        f"budget is {_human_bytes(budget)} ({_human_bytes(available)} available "
        f"x memory_fraction). Refusing rather than risking an OOM kill mid-run. "
        f"Options: " + "; ".join(remedies) + "."
    )
    return ResourceRefusal(
        message,
        required_bytes=int(required),
        budget_bytes=int(budget),
        available_bytes=int(available),
        remedies=remedies,
    )


def iter_chunks(n_rows: int, plan: ResourcePlan) -> Iterator[slice]:
    """Yield row slices honouring ``plan.chunk_rows``.

    Callers that received a chunked plan use this instead of rolling their own
    index arithmetic, so the guard's decision is actually respected.
    """
    step = max(1, plan.chunk_rows)
    for start in range(0, int(n_rows), step):
        yield slice(start, min(start + step, int(n_rows)))
