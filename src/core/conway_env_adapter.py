"""
Conway Environment Adapter

Environmental biasing of Conway rules through energy fields.
Energy levels modify rule parameters (survival/birth sets) subtly,
affecting emergence patterns without adding directional intelligence.
"""

from typing import Set, Dict, Optional, Tuple
from .conway_rules import ConwayRuleParams, SURVIVAL_SET, BIRTH_SET


class ConwayEnvironmentAdapter:
    """Adapts Conway rules based on environmental energy levels.

    This provides subtle biasing of cellular automaton behavior through
    environmental conditions, without adding agent-level intelligence or
    directional steering.
    """

    def __init__(self, default_params: Optional[ConwayRuleParams] = None):
        """Initialize environment adapter.

        Args:
            default_params: Default Conway rule parameters (standard rules if None)
        """
        self.default_params = default_params or ConwayRuleParams.standard()

        # Environmental biasing thresholds
        self.low_energy_threshold = 0.2
        self.high_energy_threshold = 0.8

    def adapt_rule_params_for_energy(self, energy_value: float) -> ConwayRuleParams:
        """Adapt Conway rule parameters based on local energy level.

        This creates subtle environmental biasing:
        - High energy: Slightly favors survival (crowded conditions)
        - Low energy: Slightly restricts birth (harsh conditions)
        - Normal energy: Standard Conway rules

        Args:
            energy_value: Local energy level (0.0 to 1.0)

        Returns:
            Adapted ConwayRuleParams for this energy level
        """
        if energy_value >= self.high_energy_threshold:
            # High energy environment: favor survival slightly
            # Allow some overcrowding to survive in "lush" conditions
            survival_set = {1, 2, 3, 4}  # Include overcrowding survival
            birth_set = {3}  # Normal birth
            reasoning = "high_energy_lush_survival"

        elif energy_value <= self.low_energy_threshold:
            # Low energy environment: restrict birth
            # Harsh conditions make birth more difficult - extinction conditions
            survival_set = {2, 3}  # Normal survival
            birth_set = set()  # No birth possible - extinction conditions
            reasoning = "low_energy_harsh_extinction"

        else:
            # Normal energy: standard Conway rules
            survival_set = SURVIVAL_SET.copy()
            birth_set = BIRTH_SET.copy()
            reasoning = "normal_energy_standard_conway"

        params = ConwayRuleParams(survival_set, birth_set)
        params._adaptation_reason = reasoning  # Store reasoning for testing

        return params

    def get_environment_description(self, energy_value: float) -> str:
        """Get human-readable description of environmental effect.

        Args:
            energy_value: Local energy level

        Returns:
            Description of how energy affects Conway rules
        """
        params = self.adapt_rule_params_for_energy(energy_value)

        survival_count = len(params.survival_set)
        birth_count = len(params.birth_set)

        if energy_value >= self.high_energy_threshold:
            return (f"High-energy lush environment: {survival_count} survival outcomes, "
                   f"{birth_count} birth outcomes (allows overcrowding)")
        elif energy_value <= self.low_energy_threshold:
            return (f"Low-energy harsh environment: {survival_count} survival outcomes, "
                   f"{birth_count} birth outcomes (birth impossible)")
        else:
            return (f"Normal energy balanced environment: {survival_count} survival outcomes, "
                   f"{birth_count} birth outcomes (standard Conway rules)")

    def create_energy_rule_map(self, energy_levels: list[float]) -> Dict[float, ConwayRuleParams]:
        """Create mapping of energy levels to Conway rule parameters.

        Useful for tabulating environmental effects across energy spectrum.

        Args:
            energy_levels: List of energy values to map

        Returns:
            Dictionary mapping energy -> adapted rule parameters
        """
        return {energy: self.adapt_rule_params_for_energy(energy)
                for energy in energy_levels}

    def validate_environmental_consistency(self) -> list[str]:
        """Validate that environmental adaptations maintain logical consistency.

        Returns:
            List of validation issues (empty if all valid)
        """
        issues = []

        # Test edge cases
        low_params = self.adapt_rule_params_for_energy(0.0)
        normal_params = self.adapt_rule_params_for_energy(0.5)
        high_params = self.adapt_rule_params_for_energy(1.0)

        # Harsh environments should have fewer birth possibilities
        if len(low_params.birth_set) >= len(normal_params.birth_set):
            issues.append("Low energy should restrict birth more than normal energy")

        # Lush environments should allow more survival possibilities
        if len(high_params.survival_set) <= len(normal_params.survival_set):
            issues.append("High energy should allow more survival than normal energy")

        # Default rules should match standard Conway
        default_survival = set(self.default_params.survival_set)
        default_birth = set(self.default_params.birth_set)
        standard_survival = set(SURVIVAL_SET)
        standard_birth = set(BIRTH_SET)

        if default_survival != standard_survival or default_birth != standard_birth:
            issues.append("Default parameters should match standard Conway rules")

        return issues


def create_standard_environment_adapter() -> ConwayEnvironmentAdapter:
    """Factory function for standard environmental adapter."""
    return ConwayEnvironmentAdapter()


def compare_energy_to_standard_conway(energy_value: float,
                                    adapter: Optional[ConwayEnvironmentAdapter] = None) -> Dict[str, any]:
    """Compare energy-adapted rules to standard Conway rules.

    Args:
        energy_value: Energy level to test
        adapter: Environment adapter (creates standard one if None)

    Returns:
        Comparison dictionary showing differences
    """
    if adapter is None:
        adapter = create_standard_environment_adapter()

    adapted_params = adapter.adapt_rule_params_for_energy(energy_value)
    standard_params = ConwayRuleParams.standard()

    return {
        "energy_level": energy_value,
        "environment_type": adapter.get_environment_description(energy_value),
        "survival_changes": {
            "standard": sorted(list(standard_params.survival_set)),
            "adapted": sorted(list(adapted_params.survival_set)),
            "added": sorted(list(adapted_params.survival_set - standard_params.survival_set)),
            "removed": sorted(list(standard_params.survival_set - adapted_params.survival_set))
        },
        "birth_changes": {
            "standard": sorted(list(standard_params.birth_set)),
            "adapted": sorted(list(adapted_params.birth_set)),
            "added": sorted(list(adapted_params.birth_set - standard_params.birth_set)),
            "removed": sorted(list(standard_params.birth_set - adapted_params.birth_set))
        },
        "behavioral_impact": "severe_restriction" if len(adapted_params.birth_set) == 0 else
                           "enhanced_survival" if len(adapted_params.survival_set) > len(standard_params.survival_set) else
                           "standard_conway"
    }


# Example usage patterns for different energy regimes
ENERGY_REGIME_EXAMPLES = {
    "extinction_zone": {
        "energy_range": (0.0, 0.2),
        "description": "Birth impossible - existing patterns decay to extinction",
        "conway_behavior": "Death spirals, only still lifes survive briefly"
    },
    "harsh_conditions": {
        "energy_range": (0.2, 0.4),
        "description": "Standard Conway rules - neutral environment",
        "conway_behavior": "Normal emergence patterns, gliders and oscillators"
    },
    "balanced_ecosystem": {
        "energy_range": (0.4, 0.8),
        "description": "Standard Conway rules - balanced dynamics",
        "conway_behavior": "Full emergence spectrum: spaceships, oscillators, still lifes"
    },
    "lush_overgrowth": {
        "energy_range": (0.8, 1.0),
        "description": "Enhanced survival allows overcrowding",
        "conway_behavior": "Dense stable patterns, slower evolution, complex forms"
    }
}


def get_energy_regime_description(energy_value: float) -> str:
    """Get descriptive regime name for an energy level.

    Args:
        energy_value: Energy value to categorize

    Returns:
        Regime description string
    """
    for regime_name, regime_info in ENERGY_REGIME_EXAMPLES.items():
        min_energy, max_energy = regime_info["energy_range"]
        if min_energy <= energy_value <= max_energy:
            return f"{regime_name}: {regime_info['description']}"

    return "unknown_regime: outside defined energy spectrum"
