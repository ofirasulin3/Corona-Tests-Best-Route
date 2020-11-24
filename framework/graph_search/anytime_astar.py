from .graph_problem_interface import *
from .astar import AStar
from .utils.timer import Timer


class AnytimeAStar(GraphProblemSolver):
    """
    This class implements a variation of the Anytime A* search algorithm.
    We execute (weighted) AStar for multiple values of `w` (=`heuristic_weight`) in a "binary-search-way" over `w`,
     and look for the best solution we can find using a limited number of expanded states for each AStar execution.
    Performs a binary search over the values [0.5, 0.95]. As always, the binary search has two bounds maintained during
     the search: low bound & high bound. The low bound means "a solution could not be found (under the #expanded-states
      constrains) for this value of w and for lower ones". The high bound means "a solution has been found for this
      value of w".
    Each iteration we perform an (weighted) AStar execution where w is the mid point (between low & high bounds) and
     update the bounds correspondingly to keep the bounds-invariant satisfied.
    This way we finds a "minimal" value of `w` for which the AStar finds a solution (for the given problem) under the
     #expanded-states constraint.
    We keep iterating until the delta between the low & high bounds gets tiny enough.
    Finally, we return the best solution found during the search, INDEPENDENTLY with the value of `w`.
    Note that the #expands is not necessarily monotony decreasing with the value of `w`, hence we don't really find the
     minimal value of `w` for which AStar stops (there actually might be two values w1<w2 such that the AStar finds
     a solution (constrained to max_nr_states_to_expand_per_iteration) using w1 but it does not find a solution
     (again, constrained to max_nr_states_to_expand_per_iteration) using w2).
    """

    solver_name = 'Anytime-A*'

    def __init__(self,
                 heuristic_function_type: HeuristicFunctionType,
                 max_nr_states_to_expand_per_iteration: int,
                 initial_high_heuristic_weight_bound: float = 0.9):
        self.heuristic_function_type = heuristic_function_type
        self.max_nr_states_to_expand_per_iteration = max_nr_states_to_expand_per_iteration
        self.initial_high_heuristic_weight_bound = initial_high_heuristic_weight_bound

    def solve_problem(self, problem: GraphProblem) -> SearchResult:
        with Timer(print_title=False) as timer:
            total_nr_expanded_states = 0
            max_nr_stored_states = 0

            acceptable_astar = AStar(heuristic_function_type=self.heuristic_function_type, heuristic_weight=0.5,
                                     max_nr_states_to_expand=self.max_nr_states_to_expand_per_iteration)
            acceptable_astar_res = acceptable_astar.solve_problem(problem)
            total_nr_expanded_states += acceptable_astar_res.nr_expanded_states
            max_nr_stored_states = max(max_nr_stored_states, acceptable_astar_res.max_nr_stored_states)
            if acceptable_astar_res.is_solution_found:
                return acceptable_astar_res._replace(
                    solver=self, nr_expanded_states=total_nr_expanded_states, solving_time=timer.elapsed)

            greedy = AStar(heuristic_function_type=self.heuristic_function_type,
                           heuristic_weight=self.initial_high_heuristic_weight_bound,
                           max_nr_states_to_expand=self.max_nr_states_to_expand_per_iteration)
            greedy_res = greedy.solve_problem(problem)
            total_nr_expanded_states += greedy_res.nr_expanded_states
            max_nr_stored_states = max(max_nr_stored_states, greedy_res.max_nr_stored_states)
            if not greedy_res.is_solution_found:
                return greedy_res._replace(
                    solver=self, nr_expanded_states=total_nr_expanded_states, solving_time=timer.elapsed)
            best_solution = greedy_res
            best_heuristic_weight = self.initial_high_heuristic_weight_bound

            # Invariant being hold during the loop:
            #   There is a solution using `high_heuristic_weight`,
            #   but there is no solution using `low_heuristic_weight`.
            low_heuristic_weight = 0.5
            high_heuristic_weight = self.initial_high_heuristic_weight_bound
            while (high_heuristic_weight - low_heuristic_weight) > 0.01:
                # TODO [Ex.45]:
                #  Complete the missing part inside this loop.
                #  Perform a binary search over the possible values of `heuristic_weight`.
                #  In each iteration, create an AStar solver with:
                #   (i)   the `heuristic_weight` set to the mid point of the current low & high binary search bound,
                #   (ii)  the `max_nr_states_to_expand` set to `self.max_nr_states_to_expand_per_iteration`,
                #   (iii) the `heuristic_function_type` set to `self.heuristic_function_type`,
                #   and solve the given problem with it.
                #  Don't forget to update `total_nr_expanded_states` and `max_nr_stored_states` (see how we've done
                #   it above).
                #  Update `low_heuristic_weight` and `high_heuristic_weight` according to the result of the AStar
                #   in order the keep the invariant (mentioned above) satisfied.
                #  You might need to use the field `is_solution_found` of the search result obtained from the AStar.
                #  Update `best_solution` and `best_heuristic_weight` if needed. `best_solution` stores the solution
                #   (SearchResult object) found with the best g-cost (use `solution_g_cost` field of SearchResult to
                #   obtain the g-cost of a solution). Update iff the current inspected solution cost < the cost of
                #   the best found solution so far.
                #  Make sure to also read the big comment in the head of this class.
                raise NotImplementedError   # TODO: remove this line!

        self.solver_name = f'{self.__class__.solver_name} (h={best_solution.solver.heuristic_function.heuristic_name}, w={best_heuristic_weight:.3f})'
        return best_solution._replace(
            solver=self, nr_expanded_states=total_nr_expanded_states, max_nr_stored_states=max_nr_stored_states,
            solving_time=timer.elapsed)
