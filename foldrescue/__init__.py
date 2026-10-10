"""foldrescue: pair-level prediction of pharmacological chaperone rescue.

Amenability is a property of a (variant, compound) PAIR. Stability predictors
answer "does this variant destabilise?"; docking answers "does this ligand
bind?". Neither answers "does this binding rescue trafficking?" -- which is the
question a clinician or an assay-prioritisation decision actually asks.

This package models the pair jointly and reports, loudly, when a query falls
outside the distribution it was trained on.

NOT CLINICAL DECISION SUPPORT. See the "What this cannot do" section of the
README before using any output for anything.
"""

__version__ = "0.1.0.dev0"

from .utils import ResourcePlan, ResourceRefusal, check_resources  # noqa: F401

__all__ = ["__version__", "check_resources", "ResourcePlan", "ResourceRefusal"]
