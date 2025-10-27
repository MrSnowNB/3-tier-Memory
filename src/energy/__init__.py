"""
CyberMesh Phase 3: Energy Fields Module

Environmental modifiers for Conway emergence patterns.
Energy fields create subtle rule biasing effects without
agent-level intelligence or directional steering.

WARNING: Advanced features like glider coupling are experimental
and disabled by default. Only use EnergyField for environmental
rule parameter modification.
"""

from .field import EnergyField, create_linear_energy_field, create_radial_energy_field

# Advanced experimental features (disabled by default)
_ENABLE_EXPERIMENTAL_COUPLING = False

if _ENABLE_EXPERIMENTAL_COUPLING:
    try:
        from .glider_coupling import EnergyGliderCoupling, EnergyGliderConfig
        _EXPERIMENTAL_FEATURES = ['EnergyGliderCoupling', 'EnergyGliderConfig']
    except ImportError:
        _EXPERIMENTAL_FEATURES = []
else:
    _EXPERIMENTAL_FEATURES = []

__version__ = "0.3.0"

# Public API: core energy field functionality + experimental features if enabled
_CORE_EXPORTS = [
    'EnergyField',
    'create_linear_energy_field',
    'create_radial_energy_field'
]

__all__ = _CORE_EXPORTS + _EXPERIMENTAL_FEATURES
