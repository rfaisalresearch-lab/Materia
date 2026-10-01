"""Driving GPAW as a real, optional, out-of-process Tier-3 solver."""

from .conversion import ConversionError, from_worker_spec, to_worker_spec
from .environment import (
    INTERPRETER_ENV_VAR,
    GPAWEnvironment,
    discover,
    reset_cache,
)
from .runner import GPAWRun, GPAWUnavailable
from .settings import (
    FUNCTIONALS,
    MODES,
    PRESETS,
    GPAWSettings,
    GPAWSettingsError,
    describe_presets,
    preset,
    validate,
)
from .solver import CAPABILITIES, GPAWSolver

__all__ = [
    "GPAWSolver", "CAPABILITIES",
    "GPAWEnvironment", "discover", "reset_cache", "INTERPRETER_ENV_VAR",
    "GPAWSettings", "GPAWSettingsError", "validate", "preset", "describe_presets",
    "PRESETS", "FUNCTIONALS", "MODES",
    "GPAWRun", "GPAWUnavailable",
    "to_worker_spec", "from_worker_spec", "ConversionError",
]
