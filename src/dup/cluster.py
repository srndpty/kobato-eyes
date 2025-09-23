"""Cluster duplicate matches into connected components."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable

from dup.refine import RefinedMatch


@dataclass
class Cluster:
    representative: int
    members: list[int]
    matches: list[RefinedMatch]


class ClusterBuilder:
    """Build clusters of duplicates using refined matches."""

    def build(self, matches: Iterable[RefinedMatch]) -> list[Cluster]:
        match_list = [m for m in matches if m.is_duplicate]
        if not match_list:
            return []

        parent: Dict[int, int] = {}

        def find(x: int) -> int:
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a: int, b: int) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a == root_b:
                return
            if root_a < root_b:
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

        for match in match_list:
            union(match.file_id_a, match.file_id_b)

        groups: Dict[int, list[int]] = defaultdict(list)
        for node in parent:
            groups[find(node)].append(node)

        cluster_matches: Dict[int, list[RefinedMatch]] = defaultdict(list)
        for match in match_list:
            root = find(match.file_id_a)
            cluster_matches[root].append(match)

        clusters: list[Cluster] = []
        for root, members in groups.items():
            members_sorted = sorted(members)
            representative = members_sorted[0]
            clusters.append(
                Cluster(
                    representative=representative,
                    members=members_sorted,
                    matches=cluster_matches[root],
                )
            )

        clusters.sort(key=lambda c: c.representative)
        return clusters


__all__ = ["Cluster", "ClusterBuilder"]
