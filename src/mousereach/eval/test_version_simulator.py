"""
Quick tests for version_simulator module.

Run with: python -m pytest test_version_simulator.py -v
Or just: python test_version_simulator.py
"""

from version_simulator import VersionFilter, apply_version_filter, VERSION_FILTERS


def test_version_filters_defined():
    """All expected versions are defined."""
    expected = ["v1.0.0", "v2.0.0", "v3.0.0", "v3.3.0", "v3.4.0"]
    for version in expected:
        assert version in VERSION_FILTERS, f"Missing version: {version}"


def test_v1_filter():
    """v1.0.0 only keeps reaches with extent > 5."""
    filter_v1 = VERSION_FILTERS["v1.0.0"]

    reaches = [
        {"max_extent_pixels": 10, "duration_frames": 5},  # KEEP
        {"max_extent_pixels": 6, "duration_frames": 5},   # KEEP
        {"max_extent_pixels": 5, "duration_frames": 5},   # DROP (not > 5)
        {"max_extent_pixels": 3, "duration_frames": 5},   # DROP
        {"max_extent_pixels": -2, "duration_frames": 5},  # DROP
    ]

    filtered = apply_version_filter(reaches, filter_v1)
    assert len(filtered) == 2, f"Expected 2 reaches, got {len(filtered)}"
    assert filtered[0]["max_extent_pixels"] == 10
    assert filtered[1]["max_extent_pixels"] == 6


def test_v2_filter():
    """v2.0.0 requires positive extent only."""
    filter_v2 = VERSION_FILTERS["v2.0.0"]

    reaches = [
        {"max_extent_pixels": 10, "duration_frames": 5},  # KEEP
        {"max_extent_pixels": 1, "duration_frames": 5},   # KEEP
        {"max_extent_pixels": 0, "duration_frames": 5},   # DROP (not > 0)
        {"max_extent_pixels": -2, "duration_frames": 5},  # DROP
    ]

    filtered = apply_version_filter(reaches, filter_v2)
    assert len(filtered) == 2, f"Expected 2 reaches, got {len(filtered)}"
    assert all(r["max_extent_pixels"] > 0 for r in filtered)


def test_v3_0_filter():
    """v3.0.0 requires extent >= 5 AND duration >= 10."""
    filter_v3 = VERSION_FILTERS["v3.0.0"]

    reaches = [
        {"max_extent_pixels": 10, "duration_frames": 15},  # KEEP
        {"max_extent_pixels": 5, "duration_frames": 10},   # KEEP
        {"max_extent_pixels": 10, "duration_frames": 5},   # DROP (duration < 10)
        {"max_extent_pixels": 3, "duration_frames": 15},   # DROP (extent < 5)
        {"max_extent_pixels": -2, "duration_frames": 15},  # DROP (extent < 5)
    ]

    filtered = apply_version_filter(reaches, filter_v3)
    assert len(filtered) == 2, f"Expected 2 reaches, got {len(filtered)}"
    assert filtered[0]["max_extent_pixels"] == 10
    assert filtered[0]["duration_frames"] == 15
    assert filtered[1]["max_extent_pixels"] == 5
    assert filtered[1]["duration_frames"] == 10


def test_v3_3_filter_bug():
    """v3.3.0 BUG: drops reaches with negative extent (extent >= 0)."""
    filter_v3_3 = VERSION_FILTERS["v3.3.0"]

    reaches = [
        {"max_extent_pixels": 10, "duration_frames": 5},  # KEEP
        {"max_extent_pixels": 5, "duration_frames": 5},   # KEEP
        {"max_extent_pixels": 0, "duration_frames": 5},   # KEEP
        {"max_extent_pixels": -2, "duration_frames": 5},  # DROP (the bug!)
        {"max_extent_pixels": -5, "duration_frames": 5},  # DROP (the bug!)
    ]

    filtered = apply_version_filter(reaches, filter_v3_3)
    assert len(filtered) == 3, f"Expected 3 reaches, got {len(filtered)}"
    # This is the BUG - it drops valid negative extent reaches


def test_v3_4_filter():
    """v3.4.0 (current) keeps all reaches regardless of extent."""
    filter_v3_4 = VERSION_FILTERS["v3.4.0"]

    reaches = [
        {"max_extent_pixels": 10, "duration_frames": 5},   # KEEP
        {"max_extent_pixels": 5, "duration_frames": 5},    # KEEP
        {"max_extent_pixels": 0, "duration_frames": 5},    # KEEP
        {"max_extent_pixels": -2, "duration_frames": 5},   # KEEP
        {"max_extent_pixels": -15, "duration_frames": 5},  # KEEP
        {"max_extent_pixels": 10, "duration_frames": 1},   # DROP (duration < 2)
    ]

    filtered = apply_version_filter(reaches, filter_v3_4)
    assert len(filtered) == 5, f"Expected 5 reaches, got {len(filtered)}"
    # Only duration filter applies


def test_bug_impact():
    """Demonstrate the impact of v3.3.0 bug on typical data."""
    # Simulate typical ground truth data based on human annotations
    # Analysis showed many valid reaches have negative extent (-2 to -15px)
    typical_reaches = [
        # Extended reaches (cross slit line)
        {"max_extent_pixels": 15, "duration_frames": 12},
        {"max_extent_pixels": 8, "duration_frames": 10},
        {"max_extent_pixels": 10, "duration_frames": 8},
        # Approach reaches (don't cross slit line but valid)
        {"max_extent_pixels": -2, "duration_frames": 8},
        {"max_extent_pixels": -5, "duration_frames": 6},
        {"max_extent_pixels": -3, "duration_frames": 7},
        {"max_extent_pixels": -8, "duration_frames": 5},
        {"max_extent_pixels": -12, "duration_frames": 6},
    ]

    # v3.4.0 (current) - finds all
    filter_current = VERSION_FILTERS["v3.4.0"]
    current_results = apply_version_filter(typical_reaches, filter_current)

    # v3.3.0 (bug) - drops negative extent
    filter_bug = VERSION_FILTERS["v3.3.0"]
    bug_results = apply_version_filter(typical_reaches, filter_bug)

    # Calculate impact
    total_reaches = len(current_results)
    detected_by_bug = len(bug_results)
    missed_by_bug = total_reaches - detected_by_bug

    print(f"\nBug Impact Analysis:")
    print(f"  Total valid reaches: {total_reaches}")
    print(f"  Detected by v3.3.0: {detected_by_bug}")
    print(f"  Missed by v3.3.0: {missed_by_bug}")
    print(f"  Recall: {detected_by_bug / total_reaches:.1%}")

    # The bug should miss ~62.5% of reaches in this example
    assert detected_by_bug == 3, f"Expected bug to find 3 reaches, got {detected_by_bug}"
    assert missed_by_bug == 5, f"Expected bug to miss 5 reaches, got {missed_by_bug}"


def test_extent_distribution():
    """Show extent distribution that motivated the v3.4.0 fix."""
    # Based on actual ground truth analysis
    extent_examples = {
        "extended (>5px)": [10, 15, 8, 12, 20],
        "short extension (0-5px)": [2, 3, 4, 1],
        "approach (-5 to 0px)": [-2, -3, -4, -1],
        "deep approach (<-5px)": [-8, -12, -15, -10]
    }

    print("\n\nExtent Distribution from Ground Truth:")
    print("="*60)

    for category, extents in extent_examples.items():
        print(f"\n{category}:")
        print(f"  Example extents: {extents}")

        # Show what each version would detect
        for version, filter_obj in VERSION_FILTERS.items():
            test_reaches = [{"max_extent_pixels": e, "duration_frames": 10} for e in extents]
            detected = apply_version_filter(test_reaches, filter_obj)
            detect_rate = len(detected) / len(extents)

            print(f"  {version}: {len(detected)}/{len(extents)} ({detect_rate:.0%})")


if __name__ == "__main__":
    # Run tests
    print("Testing version_simulator module...")
    print("="*60)

    test_version_filters_defined()
    print("✓ All version filters defined")

    test_v1_filter()
    print("✓ v1.0.0 filter works")

    test_v2_filter()
    print("✓ v2.0.0 filter works")

    test_v3_0_filter()
    print("✓ v3.0.0 filter works")

    test_v3_3_filter_bug()
    print("✓ v3.3.0 filter (bug) works as expected")

    test_v3_4_filter()
    print("✓ v3.4.0 filter (current) works")

    test_bug_impact()
    print("✓ Bug impact analysis complete")

    test_extent_distribution()
    print("\n✓ All tests passed!")
