"""
configs.py — declarative experiment arms + the OVAT one-variable guard.

An ArmConfig fully describes one pipeline: which chunking, which embedder, which
retrieval method, and the three k's. Each stage produces a list of arms that
differ from each other in EXACTLY one dimension; ``check_ovat`` enforces that so
a stage can never silently become a confounded multi-variable comparison (the
mistake that invalidated the previous P1-P4 experiment).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

# ── Allowed values per dimension (kept as plain strings so configs are JSON/CSV friendly) ──
CHUNKING_STRATEGIES = ("baseline", "breadcrumb", "per_group", "parent_child")
EMBEDDERS = ("gemini-embedding-2", "neodictabert", "bge-m3")
RETRIEVAL_METHODS = (
    "dense",            # B0: dense only
    "bm25_rrf",         # B1a: dense + BM25 via RRF
    "sparse_rrf",       # B1b: dense + learned-sparse (bge-m3 / SPLADE) via RRF
    "rerank_ce",        # B2a: B1-winner + cross-encoder rerank
    "rerank_colbert",   # B2b: B1-winner + ColBERT-v2 late interaction
    "rerank_ce_qe",     # B3: B2-winner + Hebrew query expansion
)

# The fixed constant embedder for Stages A, B and K (see plan section 3).
FIXED_EMBEDDER = "gemini-embedding-2"

# Pipeline dimensions that make two arms "different". arm_id / notes are metadata.
_PIPELINE_FIELDS = ("chunking", "embedding", "retrieval", "pool_k", "rerank_k", "context_k", "enrich")


@dataclass(frozen=True)
class ArmConfig:
    """One fully-specified retrieval pipeline."""
    arm_id: str
    chunking: str = "baseline"
    embedding: str = FIXED_EMBEDDER
    retrieval: str = "dense"
    pool_k: int = 30      # candidates pulled from first-stage retrieval
    rerank_k: int = 20    # candidates passed to the reranker
    context_k: int = 5    # final chunks handed to the generator / scored for recall
    enrich: bool = True   # attach structured metadata + URL-preserving dedup
    notes: str = ""

    def validate(self) -> None:
        if self.chunking not in CHUNKING_STRATEGIES:
            raise ValueError(f"unknown chunking '{self.chunking}'")
        if self.embedding not in EMBEDDERS:
            raise ValueError(f"unknown embedding '{self.embedding}'")
        if self.retrieval not in RETRIEVAL_METHODS:
            raise ValueError(f"unknown retrieval '{self.retrieval}'")
        if not (self.context_k <= self.rerank_k <= self.pool_k):
            raise ValueError(
                f"k ordering must satisfy context_k<=rerank_k<=pool_k, got "
                f"{self.context_k}/{self.rerank_k}/{self.pool_k}"
            )

    def pipeline_signature(self) -> dict:
        return {f: getattr(self, f) for f in _PIPELINE_FIELDS}

    def as_row(self) -> dict:
        row = {"arm_id": self.arm_id, "notes": self.notes}
        row.update(self.pipeline_signature())
        return row


def diff_fields(a: ArmConfig, b: ArmConfig) -> set[str]:
    """Pipeline dimensions in which two arms differ (ignores arm_id/notes)."""
    return {f for f in _PIPELINE_FIELDS if getattr(a, f) != getattr(b, f)}


def check_ovat(arms: list[ArmConfig], varying: set[str] | str) -> None:
    """Raise ValueError unless every arm differs from the first only within ``varying``.

    ``varying`` is the set of dimensions a stage is allowed to change (e.g.
    {"chunking"} for Stage A, or {"pool_k","rerank_k","context_k"} for Stage K).
    This is the guard that keeps each stage a clean single-variable ablation.
    """
    if isinstance(varying, str):
        varying = {varying}
    if not arms:
        raise ValueError("no arms to check")
    for arm in arms:
        arm.validate()
    baseline = arms[0]
    for arm in arms[1:]:
        changed = diff_fields(baseline, arm)
        illegal = changed - varying
        if illegal:
            raise ValueError(
                f"OVAT violation: arm '{arm.arm_id}' differs from baseline "
                f"'{baseline.arm_id}' in disallowed field(s) {sorted(illegal)}; "
                f"this stage may only vary {sorted(varying)}."
            )


# ── Stage arm builders ──────────────────────────────────────────────────────
# Winners from earlier stages are passed in, so the locked context is explicit
# and the OVAT guard can confirm only the intended dimension changes.

def stage_a_arms() -> list[ArmConfig]:
    """Stage A: vary chunking only (embedding + retrieval fixed)."""
    base = dict(embedding=FIXED_EMBEDDER, retrieval="dense", enrich=True)
    arms = [
        ArmConfig(arm_id="A0_baseline", chunking="baseline", **base),
        ArmConfig(arm_id="A1_breadcrumb", chunking="breadcrumb", **base),
        ArmConfig(arm_id="A2_per_group", chunking="per_group", **base),
        ArmConfig(arm_id="A3_parent_child", chunking="parent_child", **base),
    ]
    check_ovat(arms, varying="chunking")
    return arms


def stage_b_arms(winning_chunking: str) -> list[ArmConfig]:
    """Stage B: vary retrieval only (chunking = Stage A winner, embedding fixed)."""
    base = dict(chunking=winning_chunking, embedding=FIXED_EMBEDDER, enrich=True)
    arms = [
        ArmConfig(arm_id="B0_dense", retrieval="dense", **base),
        ArmConfig(arm_id="B1a_bm25_rrf", retrieval="bm25_rrf", **base),
        ArmConfig(arm_id="B1b_sparse_rrf", retrieval="sparse_rrf", **base),
        ArmConfig(arm_id="B2a_rerank_ce", retrieval="rerank_ce", **base),
        ArmConfig(arm_id="B2b_rerank_colbert", retrieval="rerank_colbert", **base),
        ArmConfig(arm_id="B3_rerank_ce_qe", retrieval="rerank_ce_qe", **base),
    ]
    check_ovat(arms, varying="retrieval")
    return arms


def stage_c_arms(winning_chunking: str, winning_retrieval: str) -> list[ArmConfig]:
    """Stage C: vary embedding only (chunking + retrieval locked)."""
    base = dict(chunking=winning_chunking, retrieval=winning_retrieval, enrich=True)
    arms = [
        ArmConfig(arm_id="C0_gemini2", embedding="gemini-embedding-2", **base),
        ArmConfig(arm_id="C1_neodictabert", embedding="neodictabert", **base),
        ArmConfig(arm_id="C2_bge_m3", embedding="bge-m3", **base),
    ]
    check_ovat(arms, varying="embedding")
    return arms


def stage_k_arms(locked: ArmConfig, triples: list[tuple[int, int, int]]) -> list[ArmConfig]:
    """Stage K: vary only the three k's on the otherwise-locked pipeline."""
    arms = []
    for pool_k, rerank_k, context_k in triples:
        arms.append(replace(
            locked,
            arm_id=f"K_pool{pool_k}_re{rerank_k}_ctx{context_k}",
            pool_k=pool_k, rerank_k=rerank_k, context_k=context_k,
        ))
    check_ovat([locked, *arms], varying={"pool_k", "rerank_k", "context_k"})
    return arms


# Default k sweep for Stage K (pool, rerank, context). Reranker sees the larger
# pool; final context stays small for latency + faithfulness.
DEFAULT_K_TRIPLES: list[tuple[int, int, int]] = [
    (10, 10, 3),
    (20, 15, 5),
    (30, 20, 5),
    (40, 25, 8),
    (60, 30, 8),
    (80, 40, 10),
]
