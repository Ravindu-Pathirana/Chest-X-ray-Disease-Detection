"""Cheap coverage of the T35 architecture-only benchmark grid."""
from scripts.run_t35_efficiency import GRID


def test_t35_grid_has_one_baseline_and_one_selected_per_backbone():
    assert len(GRID) == 8
    assert len({(backbone, arm) for _, backbone, arm, *_ in GRID}) == 8
    by_backbone = {}
    for _name, backbone, _arm, enabled, _gate, _reduction in GRID:
        by_backbone.setdefault(backbone, []).append(enabled)
    assert all(sorted(states) == [False, True] for states in by_backbone.values())
