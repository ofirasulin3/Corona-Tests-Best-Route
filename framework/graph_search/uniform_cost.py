from .graph_problem_interface import *
from .best_first_search import BestFirstSearch
from typing import Optional, Callable


class UniformCost(BestFirstSearch):
    """
    This class implements the Uniform Cost search algorithm.
    Uniform Cost algorithm is in the Best First Search algorithms family.
    """

    solver_name = 'UniformCost'

    def __init__(self, max_nr_states_to_expand: Optional[int] = None,
                 open_criterion: Optional[Callable[[SearchNode], bool]] = None):
        # Uniform Cost is a graph search algorithm. Hence, we use close set.
        super(UniformCost, self).__init__(
            use_close=True, max_nr_states_to_expand=max_nr_states_to_expand, open_criterion=open_criterion)

    def _open_successor_node(self, problem: GraphProblem, successor_node: SearchNode):
        if self.close.has_state(successor_node.state):
            return

        if self.open.has_state(successor_node.state):
            already_found_node_with_same_state = self.open.get_node_by_state(successor_node.state)
            if already_found_node_with_same_state.expanding_priority > successor_node.expanding_priority:
                self.open.extract_node(already_found_node_with_same_state)

        if not self.open.has_state(successor_node.state):
            self.open.push_node(successor_node)

    def _calc_node_expanding_priority(self, search_node: SearchNode) -> float:
        return search_node.g_cost
