"""Ontology domain plugins. The generic code (poc.ontology.ingest, poc.retrieval.grounded.pipeline)
runs the ingest steps and the 5-step query path; a plugin supplies everything domain-specific.
See docs/ONTOLOGY_PLUGINS.md for the interface.

The active domain comes from ONTOLOGY_DOMAIN (env override) or the active dataset adapter's
ONTOLOGY_DOMAIN attribute (financebench / docfinqa -> finance, cuad -> contracts, bharat -> gov),
defaulting to finance.
"""
import importlib
import os
from functools import lru_cache

DOMAINS = {
    "finance": "poc.ontology.domains.finance",
    "contracts": "poc.ontology.domains.contracts",
    "gov": "poc.ontology.domains.gov",
}

REQUIRED = ["NAME", "DDL", "scope_id", "doc_of", "step_base", "step_extract", "resolve", "catalogue",
            "PLAN_SYS", "execute", "evidence_block", "ANSWER_SYS"]


def active_name() -> str:
    if os.getenv("ONTOLOGY_DOMAIN"):
        return os.environ["ONTOLOGY_DOMAIN"]
    from poc import datasets

    return getattr(datasets.get(), "ONTOLOGY_DOMAIN", "finance")


@lru_cache
def get(name: str | None = None):
    name = name or active_name()
    if name not in DOMAINS:
        raise SystemExit(f"unknown ontology domain {name!r}; choose from {sorted(DOMAINS)}")
    try:
        mod = importlib.import_module(DOMAINS[name])
    except ModuleNotFoundError as e:
        if e.name != DOMAINS[name]:
            raise
        raise SystemExit(f"ontology domain plugin {name!r} is not implemented yet ({DOMAINS[name]} missing); "
                         f"see docs/ONTOLOGY_PLUGINS.md for the interface it must provide")
    missing = [a for a in REQUIRED if not hasattr(mod, a)]
    if missing:
        raise SystemExit(f"ontology domain {name!r} is missing {missing} (see docs/ONTOLOGY_PLUGINS.md)")
    return mod
