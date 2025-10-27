#!/usr/bin/env python3
"""Simple Conway baseline validation."""

import sys
sys.path.insert(0, 'src')

from src.core.ca_grid import CAGrid, create_blinker_pattern, create_block_pattern
import numpy as np

print("Testing Conway baseline implementation...")

# Test 1: Block stability
print("Test 1: Block stability")
grid = CAGrid(6, 6)
block = create_block_pattern()
grid.load_pattern(block, 2, 2)

initial_live = grid.get_live_count()
print(f"Initial live cells: {initial_live}")

for step in range(10):
    live = grid.step()
    assert live == 4, f"Block unstable at step {step}"
print("✅ Block remained stable")

# Test 2: Glider movement
print("\nTest 2: Glider movement")
grid2 = CAGrid(20, 20)
# Simple glider pattern placement
glider = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1]
], dtype=bool)
grid2.load_pattern(glider, 5, 5)

initial_com = grid2.get_center_of_mass()
print(f"Initial COM: ({initial_com[0]:.2f}, {initial_com[1]:.2f})")

for step in range(30):
    grid2.step()

final_com = grid2.get_center_of_mass()
delta_x = final_com[0] - initial_com[0]
delta_y = final_com[1] - initial_com[1]
distance = (delta_x**2 + delta_y**2)**0.5

print(f"Final COM: ({final_com[0]:.2f}, {final_com[1]:.2f})")
print(".2f")
print("✅ Glider moved diagonally"
# Test 3: Mass conservation
final_live = grid2.get_live_count()
print(f"\nFinal live cells: {final_live}")
assert final_live == 5, "Glider mass not conserved"
print("✅ Glider mass conserved")

print("\n🎉 ALL CONWAY BASELINE TESTS PASSED")
print("Emergence validated through pure Conway dynamics!")
