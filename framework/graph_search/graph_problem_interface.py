import abc
from typing import Iterator, Tuple, Optional, Type, NamedTuple, Union, Callable
from itertools import islice
from enum import Enum


"""
We define `__all__` variable in order to set which names will be
imported when writing (from another file):
>>> from framework.graph_search.graph_problem_interface import *
"""
__all__ = ['GraphProblemState', 'GraphProblem', 'GraphProblemStatesPath', 'SearchNode', 'StatesPathNode',
           'SearchResult', 'GraphProblemSolver',
           'HeuristicFunction', 'HeuristicFunctionType', 'NullHeuristic',
           'GraphProblemError', 'Cost', 'ExtendedCost', 'OperatorResult', 'StopReason']


class GraphProblemError(Exception):
    pass


class GraphProblemState(abc.ABC):
    """
    This class defines an *interface* used to represent a state of a states-space, as learnt in class.
    Notice that this is an *abstract* class. It does not represent a concrete state.
    The inheritor class must implement the abstract methods defined by this class.
    """

    @abc.abstractmethod
    def __eq__(self, other):
        """
        This is an abstract method that must be implemented by the inheritor class.
        This method is used to determine whether two given state objects represents the same state.
        Notice: Never compare floats using `==` operator!
        """
        ...

    @abc.abstractmethod
    def __hash__(self):
        """
        This is an abstract method that must be implemented by the inheritor class.
        This method is used to create a hash of a state.
        It is critical that two objects representing the same state would have the same hash!
        A common implementation might be something in the format of:
        >>> hash((self.some_field1, self.some_field2, self.some_field3))
        Notice: Do NOT give float fields to `hash()`. Otherwise the upper requirement would not met.
        """
        ...

    @abc.abstractmethod
    def __str__(self):
        """
        This is an abstract method that must be implemented by the inheritor class.
        This method is used by the printing mechanism of `SearchResult`.
        """


class ExtendedCost(abc.ABC):
    """
    Used as an interface for a cost type.
    Custom cost type is needed when a problem has multiple cost functions that
     each one of them should individually accumulated during the search.
    The `g_cost` is a single float scalar that should be eventually optimized
     by the search algorithm. The `g_cost` can be, for example, just one of the
     accumulated cost functions, or any function of these.
    """

    @abc.abstractmethod
    def get_g_cost(self) -> float: ...

    @abc.abstractmethod
    def __add__(self, other) -> 'ExtendedCost': ...


Cost = Union[float, ExtendedCost]


class OperatorResult(NamedTuple):
    successor_state: GraphProblemState
    operator_cost: Cost
    operator_name: Optional[str] = None


class StatesPathNode(NamedTuple):
    state: GraphProblemState
    last_operator_cost: Cost
    cumulative_cost: Cost
    cumulative_g_cost: Cost
    last_operator_name: Optional[str] = None


class GraphProblem(abc.ABC):
    """
    This class defines an *interface* used to represent a states-space, as learnt in class.
    Notice that this is an *abstract* class. It does not represent a concrete states-space.
    The inheritor class must implement the abstract methods defined by this class.
    By defining these abstract methods, the inheritor class represents a well-defined states-space.
    """

    """Each problem might have a name as a string. This name is used in the solution printings."""
    name: str = ''

    def __init__(self, initial_state: GraphProblemState):
        self.initial_state = initial_state

    @abc.abstractmethod
    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[OperatorResult]:
        """
        This is an abstract method that must be implemented by the inheritor class.
        This method represents the `Succ: S -> P(S)` function (as learnt in class) of the problem.
        It receives a state and iterates over the successor states.
        Notice that this is an *Iterator*. Hence it should be implemented using the `yield` keyword.
        For each successor, an object of type `OperatorResult` is yielded. This object describes the
            successor state, the cost of the applied operator and its name.
        """
        ...

    @abc.abstractmethod
    def is_goal(self, state: GraphProblemState) -> bool:
        """
        This is an abstract method that must be implemented by the inheritor class.
        It receives a state and returns whether this state is a goal.
        """
        ...

    def get_zero_cost(self) -> Cost:
        """
        The search algorithm should be able to use a zero cost object in order to
         initialize the cumulative cost.
        The default implementation assumes the problem uses `float` cost, and hence
         simply returns scalar value of `0`.
        When using an extended cost type (and not just float scalar), this method
         should be overridden and return an instance (of the extended cost type)
         with a "zero cost" meaning.
        """
        return 0.0

    def solution_additional_str(self, result: 'SearchResult') -> str:
        """
        This method may be overridden by the inheritor class.
        It is used to enhance the printing method of a found solution.
        We implemented it wherever needed - you do not have to care about it.
        """
        return ''


class GraphProblemStatesPath(Tuple[StatesPathNode]):
    """
    This class represents a path of states.
    It is just a tuple of GraphProblemState objects.
    We define a dedicated class in order to implement the string formatting method.
    """

    def __eq__(self, other):
        assert isinstance(other, GraphProblemStatesPath)
        if len(other) != len(self):
            return False
        return all(s1 == s2 for s1, s2 in zip(self, other))

    def __str__(self):
        if len(self) == 0:
            return '[]'
        return '[' + str(self[0].state) + \
               ''.join(f'  ={"" if action.last_operator_name is None else f"=({action.last_operator_name})="}=>  ' + str(action.state)
                       for action in islice(self, 1, None))\
               + ']'

        # return '[' + (' ==> '.join(str(state) for state in self)) + ']'


class SearchNode:
    """
    An object of type `SearchNode` represent a node created by a search algorithm.
    A node basically has a state that it represents, and potentially a parent node.
    A node may also have its cost, the cost of the operator performed to reach this node,
    and the f-score of this node (expanding_priority) when needed.
    """

    def __init__(self, state: GraphProblemState,
                 parent_search_node: Optional['SearchNode'] = None,
                 operator_cost: Cost = 0.0, operator_name: Optional[str] = None,
                 expanding_priority: Optional[float] = None):
        self.state: GraphProblemState = state
        self.parent_search_node: SearchNode = parent_search_node
        self.operator_cost: Cost = operator_cost
        self.operator_name: Optional[str] = operator_name
        self.expanding_priority: Optional[float] = expanding_priority

        self.cost: Cost = operator_cost
        if self.parent_search_node is not None:
            self.cost += self.parent_search_node.cost

    def traverse_back_to_root(self) -> Iterator['SearchNode']:
        """
        This is an iterator. It iterates over the nodes in the path
        starting from this node and ending in the root node.
        """
        node = self
        while node is not None:
            assert(isinstance(node, SearchNode))
            yield node
            node = node.parent_search_node

    def make_states_path(self) -> GraphProblemStatesPath:
        """
        :return: A path of *states* represented by the nodes
        in the path from the root to this node.
        """
        path = [StatesPathNode(state=node.state, last_operator_cost=node.operator_cost,
                               cumulative_cost=node.cost, cumulative_g_cost=node.g_cost,
                               last_operator_name=node.operator_name)
                for node in self.traverse_back_to_root()]
        path.reverse()
        return GraphProblemStatesPath(path)

    @property
    def g_cost(self) -> float:
        if isinstance(self.cost, float):
            return self.cost
        else:
            assert isinstance(self.cost, ExtendedCost)
            return self.cost.get_g_cost()


class StopReason(Enum):
    CompletedRunSuccessfully = 'CompletedRunSuccessfully'
    ExceededMaxNrIteration = 'ExceededMaxNrIteration'
    ExceededMaxNrStatesToExpand = 'ExceededMaxNrStatesToExpand'


class SearchResult(NamedTuple):
    """
    It is the type of the object that is returned by `solver.solve_problem()`.
    It stores the results of the search.
    """

    """The solver that generated this result."""
    solver: 'GraphProblemSolver'
    """The problem that the solver has attempted to solve."""
    problem: GraphProblem
    """The number of expanded states during the search."""
    nr_expanded_states: int
    """The maximum number of states that have been stored in open & close states during the search."""
    max_nr_stored_states: int
    """The time (in seconds) took to solve."""
    solving_time: Optional[float] = None
    """States path (including the applied operators) from the initial state to the final found goal state.
            Set to `None` if no result had been found."""
    solution_path: Optional[GraphProblemStatesPath] = None
    """Number of iterations (for an iterative algorithm like iterative-deepening)"""
    nr_iterations: Optional[int] = None
    """Whether the search ended as expected or stopped because of end of resources."""
    stop_reason: StopReason = StopReason.CompletedRunSuccessfully

    def __str__(self):
        """
        Enhanced string formatting for the search result.
        """

        res_str = f'{self.problem.name: <35}' \
                  f'   {self.solver.solver_name: <27}'

        if self.solving_time is not None:
            res_str += f'   time: {self.solving_time:6.2f}'

        res_str += f'   #dev: {self.nr_expanded_states: <5}' \
                   f'   |space|: {self.max_nr_stored_states: <6}'

        if self.nr_iterations is not None:
            res_str += f'   #iter: {self.nr_iterations: <3}'

        if self.stop_reason != StopReason.CompletedRunSuccessfully:
            assert not self.is_solution_found
            StopReasonToDescriptionMapping = {
                StopReason.ExceededMaxNrStatesToExpand: 'Exceeded max number of states to expand!',
                StopReason.ExceededMaxNrIteration: 'Exceeded max number of iterations!'
            }
            return res_str + '   ' + StopReasonToDescriptionMapping[self.stop_reason]

        # no solution found by solver
        if not self.is_solution_found:
            return res_str + '   NO SOLUTION FOUND !!!'

        res_str += f'   total_g_cost: {self.solution_g_cost:11.5f}'
        if not isinstance(self.solution_cost, float):
            res_str += f'   total_cost: {self.solution_cost}'
        res_str += f'   |path|: {len(self.solution_path)-1: <3}' \
                   f'   path: {str(self.solution_path)}'

        additional_str = self.problem.solution_additional_str(self)
        if additional_str:
            res_str += '   ' + additional_str

        return res_str

    @property
    def is_solution_found(self) -> bool:
        return self.solution_path is not None

    @property
    def solution_cost(self) -> Optional[Cost]:
        return None if self.solution_path is None else self.solution_path[-1].cumulative_cost

    @property
    def solution_g_cost(self) -> Optional[Cost]:
        return None if self.solution_path is None else self.solution_path[-1].cumulative_g_cost

    @property
    def solution_final_state(self) -> Optional[GraphProblemState]:
        return None if self.solution_path is None else self.solution_path[-1].state


class GraphProblemSolver(abc.ABC):
    """
    This class is simply just an interface for graph search algorithms.
    Each search algorithm that we are going to implement will inherit
    from this class and implement the `solve_problem()` method.
    """

    """The solver name is used when printing the search results.
    It may be overridden by the inheritor algorithm."""
    solver_name: str = 'GraphProblemSolver'

    @abc.abstractmethod
    def solve_problem(self, problem: GraphProblem) -> SearchResult:
        ...


class HeuristicFunction(abc.ABC):
    """
    This is an interface for a heuristic function.
    Each implementation of a concrete heuristic function inherits from this class.
    """

    """Used by the solution printings.
    Might be overridden by the inheritor heuristic."""
    heuristic_name = ''

    def __init__(self, problem: GraphProblem):
        self.problem = problem

    @abc.abstractmethod
    def estimate(self, state: GraphProblemState) -> float:
        """
        Calculates and returns the heuristic value for a given state.
        This is an abstract method that must be implemented by the inheritor.
        """
        ...


"""Search algorithm which uses a heuristic may receive in their
constructor the type of the heuristic to use, rather than an
already-created instance of the heuristic."""
HeuristicFunctionType = Union[Type[HeuristicFunction], Callable[[GraphProblem], HeuristicFunction]]


class NullHeuristic(HeuristicFunction):
    """
    This is a simple implementation of the null heuristic.
    It might be used with A* for a sanity-check (A* should
    behave exactly like UniformCost in that case).
    """

    heuristic_name = '0'

    def estimate(self, state: GraphProblemState) -> float:
        return 0

