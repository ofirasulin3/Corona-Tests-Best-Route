from typing import Dict, FrozenSet
from framework import Junction


class CachedAirDistanceCalculator:
    def __init__(self):
        self.junctions_pair_to_cached_air_distances_mapping: Dict[FrozenSet[Junction], float] = {}

    def get_air_distance_between_junctions(self, junction1: Junction, junction2: Junction) -> float:
        key = frozenset((junction1, junction2))
        if key not in self.junctions_pair_to_cached_air_distances_mapping:
            self.junctions_pair_to_cached_air_distances_mapping[key] = junction1.calc_air_distance_from(junction2)
        return self.junctions_pair_to_cached_air_distances_mapping[key]
