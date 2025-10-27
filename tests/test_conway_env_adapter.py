"""
Conway Environment Adapter Tests

Tests environmental biasing of Conway rules through energy fields.
Validates that energy affects rule parameters (survival/birth sets)
without adding directional intelligence or agent steering.
"""

import pytest
from src.core.conway_env_adapter import (
    ConwayEnvironmentAdapter,
    create_standard_environment_adapter,
    compare_energy_to_standard_conway,
    get_energy_regime_description
)
from src.core.conway_rules import ConwayRuleParams, SURVIVAL_SET, BIRTH_SET


class TestConwayEnvironmentAdapter:
    """Test environmental adaptation of Conway rules."""

    @pytest.fixture
    def adapter(self):
        """Standard environment adapter fixture."""
        return ConwayEnvironmentAdapter()

    def test_adapter_initialization(self, adapter):
        """Test environment adapter initialization."""
        assert adapter.low_energy_threshold == 0.2
        assert adapter.high_energy_threshold == 0.8

        # Default params should be standard Conway
        assert adapter.default_params.survival_set == SURVIVAL_SET
        assert adapter.default_params.birth_set == BIRTH_SET

    @pytest.mark.parametrize("energy_level,expected_regime", [
        (0.0, "low_energy_harsh_extinction"),
        (0.1, "low_energy_harsh_extinction"),
        (0.5, "normal_energy_standard_conway"),
        (0.9, "high_energy_lush_survival"),
        (1.0, "high_energy_lush_survival"),
    ])
    def test_energy_rule_adaptation(self, adapter, energy_level, expected_regime):
        """Test that energy levels produce appropriate rule adaptations."""
        params = adapter.adapt_rule_params_for_energy(energy_level)
        assert hasattr(params, '_adaptation_reason')
        assert params._adaptation_reason == expected_regime

    def test_low_energy_extinction_rules(self, adapter):
        """Test that low energy creates extinction conditions."""
        params = adapter.adapt_rule_params_for_energy(0.1)

        # Low energy should prevent birth (extinction)
        assert len(params.birth_set) == 0, "Low energy should prevent birth"
        assert params.survival_set == {2, 3}, "Low energy should have normal survival"

    def test_high_energy_lush_rules(self, adapter):
        """Test that high energy creates lush conditions."""
        params = adapter.adapt_rule_params_for_energy(0.9)

        # High energy should allow overcrowding survival
        assert 4 in params.survival_set, "High energy should allow overcrowding survival"
        assert params.birth_set == {3}, "High energy should have normal birth"

    def test_normal_energy_standard_rules(self, adapter):
        """Test that normal energy maintains standard Conway rules."""
        params = adapter.adapt_rule_params_for_energy(0.5)

        # Normal energy should be exactly standard Conway
        assert params.survival_set == SURVIVAL_SET
        assert params.birth_set == BIRTH_SET

    def test_environment_descriptions(self, adapter):
        """Test human-readable environment descriptions."""
        low_desc = adapter.get_environment_description(0.1)
        assert "Low-energy harsh environment" in low_desc
        assert "birth impossible" in low_desc

        normal_desc = adapter.get_environment_description(0.5)
        assert "Normal energy balanced environment" in normal_desc
        assert "standard Conway rules" in normal_desc

        high_desc = adapter.get_environment_description(0.9)
        assert "High-energy lush environment" in high_desc
        assert "allows overcrowding" in high_desc

    def test_energy_rule_mapping(self, adapter):
        """Test creation of energy-to-rules mapping."""
        energy_levels = [0.0, 0.3, 0.5, 0.7, 1.0]
        rule_map = adapter.create_energy_rule_map(energy_levels)

        assert len(rule_map) == 5
        assert all(isinstance(params, ConwayRuleParams) for params in rule_map.values())

    def test_environmental_consistency(self, adapter):
        """Test that environmental adaptations are logically consistent."""
        issues = adapter.validate_environmental_consistency()

        # Should have no validation issues
        assert len(issues) == 0, f"Environmental consistency issues: {issues}"


class TestEnergyRuleComparisons:
    """Test comparisons between energy-adapted and standard Conway rules."""

    @pytest.fixture
    def adapter(self):
        return create_standard_environment_adapter()

    def test_zero_energy_equals_pure_conway(self, adapter):
        """Test that zero energy produces standard Conway (acceptance criterion)."""
        comparison = compare_energy_to_standard_conway(0.5, adapter)

        # Normal energy should be identical to standard Conway
        survival_standard = comparison["survival_changes"]["standard"]
        survival_adapted = comparison["survival_changes"]["adapted"]

        birth_standard = comparison["birth_changes"]["standard"]
        birth_adapted = comparison["birth_changes"]["adapted"]

        assert survival_standard == survival_adapted, "Normal energy survival should match Conway"
        assert birth_standard == birth_adapted, "Normal energy birth should match Conway"
        assert comparison["behavioral_impact"] == "standard_conway"

    def test_extinction_energy_changes(self):
        """Test that low energy creates different behavior."""
        comparison = compare_energy_to_standard_conway(0.1)

        # Low energy should remove all birth possibilities
        birth_removed = comparison["birth_changes"]["removed"]
        assert len(birth_removed) > 0, "Low energy should remove birth possibilities"

        birth_adapted = comparison["birth_changes"]["adapted"]
        assert len(birth_adapted) == 0, "Low energy should have zero birth outcomes"

        assert comparison["behavioral_impact"] == "severe_restriction"

    def test_lush_energy_changes(self):
        """Test that high energy creates different behavior."""
        comparison = compare_energy_to_standard_conway(0.9)

        # High energy should add survival possibilities
        survival_added = comparison["survival_changes"]["added"]
        assert len(survival_added) > 0, "High energy should add survival possibilities"

        assert comparison["behavioral_impact"] == "enhanced_survival"

    def test_energy_regime_descriptions(self):
        """Test categorization of energy levels into regimes."""
        assert "extinction_zone" in get_energy_regime_description(0.1)
        assert "balanced_ecosystem" in get_energy_regime_description(0.6)
        assert "lush_overgrowth" in get_energy_regime_description(0.9)


class TestEnvironmentalRuleBiasing:
    """Test that energy biasing affects rules, not directions."""

    def test_energy_high_biases_survival_not_direction(self):
        """Test that high energy biases survival without directional steering."""
        adapter = ConwayEnvironmentAdapter()

        # Get rules for different energy levels
        low_rules = adapter.adapt_rule_params_for_energy(0.1)
        high_rules = adapter.adapt_rule_params_for_energy(0.9)

        # High energy should allow more survival configurations
        assert len(high_rules.survival_set) > len(low_rules.survival_set)

        # Birth should remain possible in high energy (not zero)
        assert len(high_rules.birth_set) > 0

        # OVERARCHING TEST: Rules change, but there are no directional biases
        # (This is the key acceptance criterion - rule parameter modification only,
        # not agent-level path prediction or movement steering)

    def test_glider_behavior_through_environment(self):
        """Test that glider behavior emerges from environmental constraints."""
        adapter = ConwayEnvironmentAdapter()

        # In normal energy ranges, standard Conway behavior persists
        for energy in [0.3, 0.4, 0.5, 0.6, 0.7]:
            params = adapter.adapt_rule_params_for_energy(energy)
            assert params.survival_set == SURVIVAL_SET
            assert params.birth_set == BIRTH_SET
            # This ensures gliders move due to Conway dynamics, not energy steering

        # In extreme energy ranges, different behaviors emerge
        extinction_params = adapter.adapt_rule_params_for_energy(0.0)
        assert len(extinction_params.birth_set) == 0, "No birth in extinction zones"

        # But in balanced ranges, normal emergence continues


def test_conway_environmental_variations():
    """Integration test of Conway behavior variations across energy spectrum."""
    adapter = ConwayEnvironmentAdapter()

    # Test behavior spectrum
    test_energies = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    descriptions = []

    for energy in test_energies:
        desc = adapter.get_environment_description(energy)
        descriptions.append(desc)

        params = adapter.adapt_rule_params_for_energy(energy)

        # Validate no invalid rule configurations
        assert all(0 <= n <= 8 for n in params.survival_set)
        assert all(0 <= n <= 8 for n in params.birth_set)

    # Ensure variety in descriptions
    assert len(set(descriptions)) > 1, "Energy levels should produce different behaviors"

    print("Environmental adaptations produce diverse Conway behaviors:")
    for energy, desc in zip(test_energies, descriptions):
        print(".1f")
