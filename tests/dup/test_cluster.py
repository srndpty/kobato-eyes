"""Tests for duplicate clustering."""

from __future__ import annotations

from dup.cluster import ClusterBuilder
from dup.refine import RefinedMatch


def test_cluster_builder_groups_connected_components() -> None:
    matches = [
        RefinedMatch(file_id_a=1, file_id_b=2, ssim=0.95, orb_ratio=0.2, is_duplicate=True, reason="ssim"),
        RefinedMatch(file_id_a=2, file_id_b=3, ssim=0.93, orb_ratio=0.15, is_duplicate=True, reason="ssim"),
        RefinedMatch(file_id_a=4, file_id_b=5, ssim=0.91, orb_ratio=0.16, is_duplicate=True, reason="ssim"),
        RefinedMatch(file_id_a=3, file_id_b=5, ssim=0.5, orb_ratio=0.05, is_duplicate=False, reason="below"),
    ]

    builder = ClusterBuilder()
    clusters = builder.build(matches)

    assert len(clusters) == 2
    cluster_members = [cluster.members for cluster in clusters]
    assert [1, 2, 3] in cluster_members
    assert [4, 5] in cluster_members
