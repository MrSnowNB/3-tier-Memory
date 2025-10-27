"""
CyberMesh Phase 3: Energy Fields Module

Dynamic energy gradients that create emergent routing behavior
in the cellular automaton substrate.
"""

from .field import EnergyField, create_linear_energy_field, create_radial_energy_field

__version__ = "0.3.0"
__all__ = [
    'EnergyField',
    'create_linear_energy_field',
    'create_radial_energy_field'
]
