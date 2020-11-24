from framework import *

from typing import Iterator
from dataclasses import dataclass


__all__ = ['MapState', 'MapProblem']


@dataclass(frozen=True)
class MapState(GraphProblemState):
    """
    StreetsMap state is represents the current geographic location on the map.
    This location is defined by the junction index.
    """
    junction_id: int

    def __eq__(self, other):
        assert isinstance(other, MapState)
        return other.junction_id == self.junction_id

    def __hash__(self):
        return hash(self.junction_id)

    def __str__(self):
        return str(self.junction_id).rjust(5, ' ')


class MapProblem(GraphProblem):
    """
    Represents a problem on the streets map.
    The problem is defined by a source location on the map and a destination.
    """

    name = 'StreetsMap'

    def __init__(self, streets_map: StreetsMap, source_junction_id: int, target_junction_id: int):
        initial_state = MapState(source_junction_id)
        super(MapProblem, self).__init__(initial_state)
        self.streets_map = streets_map
        self.target_junction_id = target_junction_id
        self.name += f'(src: {source_junction_id} dst: {target_junction_id})'
    
    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[OperatorResult]:
        """
        For a given state, iterates over its successor states.
        The successor states represents the junctions to which there
         exists a road originates from the given state.
        """

        # All of the states in this problem are instances of the class `MapState`.
        assert isinstance(state_to_expand, MapState)

        # Get the junction (in the map) that is represented by the state to expand.
        junction = self.streets_map[state_to_expand.junction_id]

        for outgoing_link in junction.outgoing_links:
            if outgoing_link is not None:
                yield OperatorResult(successor_state=MapState(outgoing_link.target),
                                     operator_cost=outgoing_link.distance)

    def is_goal(self, state: GraphProblemState) -> bool:
        """
        :return: Whether a given map state represents the destination.
        """
        assert (isinstance(state, MapState))

        # You may use the problem's input parameters (stored as fields of this object by the constructor).
        return state.junction_id == self.target_junction_id
