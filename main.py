from framework import *
from problems import *

from matplotlib import pyplot as plt
import numpy as np
from typing import List, Union, Optional

# Load the streets map
streets_map = StreetsMap.load_from_csv(Consts.get_data_file_path("tlv_streets_map.csv"))

# Make sure that the whole execution is deterministic.
# This is important, because we expect to get the exact same results
# in each execution.
Consts.set_seed()


def plot_distance_and_expanded_wrt_weight_figure(
        problem_name: str,
        weights: Union[np.ndarray, List[float]],
        total_cost: Union[np.ndarray, List[float]],
        total_nr_expanded: Union[np.ndarray, List[int]]):
    """
    Use `matplotlib` to generate a figure of the distance & #expanded-nodes
     w.r.t. the weight.
    TODO [Ex.15]: Complete the implementation of this method.
    """
    weights, total_cost, total_nr_expanded = np.array(weights), np.array(total_cost), np.array(total_nr_expanded)
    assert len(weights) == len(total_cost) == len(total_nr_expanded)
    assert len(weights) > 0
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    assert is_sorted(weights)

    fig, ax1 = plt.subplots()

    # TODO: Plot the total distances with ax1. Use `ax1.plot(...)`.
    # TODO: Make this curve colored blue with solid line style.
    # TODO: Set its label to be 'Solution cost'.
    # See documentation here:
    # https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html
    # You can also Google for additional examples.
    raise NotImplementedError  # TODO: remove this line!
    p1, = ax1.plot(...)  # TODO: pass the relevant params instead of `...`.

    # ax1: Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Solution cost', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xlabel('weight')

    # Create another axis for the #expanded curve.
    ax2 = ax1.twinx()

    # TODO: Plot the total expanded with ax2. Use `ax2.plot(...)`.
    # TODO: Make this curve colored red with solid line style.
    # TODO: Set its label to be '#Expanded states'.
    raise NotImplementedError  # TODO: remove this line!
    p2, = ax2.plot(...)  # TODO: pass the relevant params instead of `...`.

    # ax2: Make the y-axis label, ticks and tick labels match the line color.
    ax2.set_ylabel('#Expanded states', color='r')
    ax2.tick_params('y', colors='r')

    curves = [p1, p2]
    ax1.legend(curves, [curve.get_label() for curve in curves])

    fig.tight_layout()
    plt.title(f'Quality vs. time for wA* \non problem {problem_name}')
    plt.show()


def run_astar_for_weights_in_range(heuristic_type: HeuristicFunctionType, problem: GraphProblem, n: int = 30,
                                   max_nr_states_to_expand: Optional[int] = 40_000,
                                   low_heuristic_weight: float = 0.5, high_heuristic_weight: float = 0.95):
    # TODO [Ex.15]:
    #  1. Create an array of `n` numbers equally spread in the segment
    #     [low_heuristic_weight, high_heuristic_weight]
    #     (including the edges). You can use `np.linspace()` for that.
    #  2. For each weight in that array run the wA* algorithm, with the
    #     given `heuristic_type` over the given problem. For each such run,
    #     if a solution has been found (res.is_solution_found), store the
    #     cost of the solution (res.solution_g_cost), the number of
    #     expanded states (res.nr_expanded_states), and the weight that
    #     has been used in this iteration. Store these in 3 lists (list
    #     for the costs, list for the #expanded and list for the weights).
    #     These lists should be of the same size when this operation ends.
    #     Don't forget to pass `max_nr_states_to_expand` to the AStar c'tor.
    #  3. Call the function `plot_distance_and_expanded_wrt_weight_figure()`
    #     with these 3 generated lists.
    raise NotImplementedError  # TODO: remove this line!


# --------------------------------------------------------------------
# ------------------------ StreetsMap Problem ------------------------
# --------------------------------------------------------------------

def toy_map_problem_experiments():
    print()
    print('Solve the map problem.')

    # Ex.10
    # TODO: Just run it and inspect the printed result.
    toy_map_problem = MapProblem(streets_map, 54, 549)
    uc = UniformCost()
    res = uc.solve_problem(toy_map_problem)
    print(res)

    # Ex.12
    # TODO: create an instance of `AStar` with the `NullHeuristic`,
    #       solve the same `toy_map_problem` with it and print the results (as before).
    # Notice: AStar constructor receives the heuristic *type* (ex: `MyHeuristicClass`),
    #         and NOT an instance of the heuristic (eg: not `MyHeuristicClass()`).
    exit()  # TODO: remove!

    # Ex.13
    # TODO: create an instance of `AStar` with the `AirDistHeuristic`,
    #       solve the same `toy_map_problem` with it and print the results (as before).
    exit()  # TODO: remove!

    # Ex.15
    # TODO:
    #  1. Complete the implementation of the function
    #     `run_astar_for_weights_in_range()` (upper in this file).
    #  2. Complete the implementation of the function
    #     `plot_distance_and_expanded_wrt_weight_figure()`
    #     (upper in this file).
    #  3. Call here the function `run_astar_for_weights_in_range()`
    #     with `AirDistHeuristic` and `toy_map_problem`.
    exit()  # TODO: remove!


# --------------------------------------------------------------------
# ---------------------------- MDA Problem ---------------------------
# --------------------------------------------------------------------

loaded_problem_inputs_by_size = {}
loaded_problems_by_size_and_opt_obj = {}


def get_mda_problem(
        problem_input_size: str = 'small',
        optimization_objective: MDAOptimizationObjective = MDAOptimizationObjective.Distance):
    if (problem_input_size, optimization_objective) in loaded_problems_by_size_and_opt_obj:
        return loaded_problems_by_size_and_opt_obj[(problem_input_size, optimization_objective)]
    assert problem_input_size in {'small', 'moderate', 'big'}
    if problem_input_size not in loaded_problem_inputs_by_size:
        loaded_problem_inputs_by_size[problem_input_size] = MDAProblemInput.load_from_file(
            f'{problem_input_size}_mda.in', streets_map)
    problem = MDAProblem(
        problem_input=loaded_problem_inputs_by_size[problem_input_size],
        streets_map=streets_map,
        optimization_objective=optimization_objective)
    loaded_problems_by_size_and_opt_obj[(problem_input_size, optimization_objective)] = problem
    return problem


def basic_mda_problem_experiments():
    print()
    print('Solve the MDA problem (small input, only distance objective, UniformCost).')

    small_mda_problem_with_distance_cost = get_mda_problem('small', MDAOptimizationObjective.Distance)

    # Ex.18
    # TODO: create an instance of `UniformCost`, solve the `small_mda_problem_with_distance_cost`
    #       with it and print the results.
    exit()  # TODO: remove!


def mda_problem_with_astar_experiments():
    print()
    print('Solve the MDA problem (moderate input, only distance objective, A*, '
          'MaxAirDist & SumAirDist & MSTAirDist heuristics).')

    moderate_mda_problem_with_distance_cost = get_mda_problem('moderate', MDAOptimizationObjective.Distance)

    # Ex.22
    # TODO: create an instance of `AStar` with the `MDAMaxAirDistHeuristic`,
    #       solve the `moderate_mda_problem_with_distance_cost` with it and print the results.
    exit()  # TODO: remove!

    # Ex.25
    # TODO: create an instance of `AStar` with the `MDASumAirDistHeuristic`,
    #       solve the `moderate_mda_problem_with_distance_cost` with it and print the results.
    exit()  # TODO: remove!

    # Ex.28
    # TODO: create an instance of `AStar` with the `MDAMSTAirDistHeuristic`,
    #       solve the `moderate_mda_problem_with_distance_cost` with it and print the results.
    exit()  # TODO: remove!


def mda_problem_with_weighted_astar_experiments():
    print()
    print('Solve the MDA problem (small & moderate input, only distance objective, wA*).')

    small_mda_problem_with_distance_cost = get_mda_problem('small', MDAOptimizationObjective.Distance)
    moderate_mda_problem_with_distance_cost = get_mda_problem('moderate', MDAOptimizationObjective.Distance)

    # Ex.30
    # TODO: Call here the function `run_astar_for_weights_in_range()`
    #       with `MDAMSTAirDistHeuristic`
    #       over the `small_mda_problem_with_distance_cost`.
    exit()  # TODO: remove!

    # Ex.30
    # TODO: Call here the function `run_astar_for_weights_in_range()`
    #       with `MDASumAirDistHeuristic`
    #       over the `moderate_mda_problem_with_distance_cost`.
    exit()  # TODO: remove!


def monetary_cost_objectives_mda_problem_experiments():
    print()
    print('Solve the MDA problem (monetary objectives).')

    small_mda_problem_with_monetary_cost = get_mda_problem('small', MDAOptimizationObjective.Monetary)
    moderate_mda_problem_with_monetary_cost = get_mda_problem('moderate', MDAOptimizationObjective.Monetary)

    # Ex.32
    # TODO: create an instance of `UniformCost`
    #       solve the `small_mda_problem_with_monetary_cost` with it and print the results.
    exit()  # TODO: remove!

    # Ex.32
    # TODO: create an instance of `UniformCost`
    #       solve the `moderate_mda_problem_with_monetary_cost` with it and print the results.
    exit()  # TODO: remove!


def multiple_objectives_mda_problem_experiments():
    print()
    print('Solve the MDA problem (moderate input, distance & tests-travel-distance objectives).')

    moderate_mda_problem_with_distance_cost = get_mda_problem('moderate', MDAOptimizationObjective.Distance)
    moderate_mda_problem_with_tests_travel_dist_cost = get_mda_problem('moderate', MDAOptimizationObjective.TestsTravelDistance)

    # Ex.35
    # TODO: create an instance of `AStar` with the `MDATestsTravelDistToNearestLabHeuristic`,
    #       solve the `moderate_mda_problem_with_tests_travel_dist_cost` with it and print the results.
    exit()  # TODO: remove!

    # Ex.38
    # TODO: Implement the algorithm A_2 described in this exercise in the assignment instructions.
    #       Create an instance of `AStar` with the `MDAMSTAirDistHeuristic`.
    #       Solve the `moderate_mda_problem_with_distance_cost` with it and store the solution's (optimal)
    #         distance cost to the variable `optimal_distance_cost`.
    #       Calculate the value (1 + eps) * optimal_distance_cost in the variable `max_distance_cost` (for eps=0.6).
    #       Create another instance of `AStar` with the `MDATestsTravelDistToNearestLabHeuristic`, and specify the
    #          param `open_criterion` (to AStar c'tor) to be the criterion mentioned in the A_2 algorithm in the
    #          assignment instructions. Use a lambda function for that. This function should receive a `node` and
    #          has to return whether to add this just-created-node to the `open` queue.
    #          Remember that in python you can pass an argument to a function's parameter by the parameter's name
    #          `some_func(argument_name=some_value)`. This becomes especially relevant when you want to leave some
    #          previous parameters with their default values and pass an argument to a parameter that is positioned
    #          elsewhere next.
    #       Solve the `moderate_mda_problem_with_tests_travel_dist_cost` with it and print the results.
    exit()  # TODO: remove!


def mda_problem_with_astar_epsilon_experiments():
    print()
    print('Solve the MDA problem (small input, distance objective, using A*eps, use non-acceptable '
          'heuristic as focal heuristic).')

    small_mda_problem_with_distance_cost = get_mda_problem('small', MDAOptimizationObjective.Distance)

    # Firstly solve the problem with AStar & MST heuristic for having a reference for #devs.
    astar = AStar(MDAMSTAirDistHeuristic)
    res = astar.solve_problem(small_mda_problem_with_distance_cost)
    print(res)

    def within_focal_h_sum_priority_function(node: SearchNode, problem: GraphProblem, solver: AStarEpsilon):
        if not hasattr(solver, '__focal_heuristic'):
            setattr(solver, '__focal_heuristic', MDASumAirDistHeuristic(problem=problem))
        focal_heuristic = getattr(solver, '__focal_heuristic')
        return focal_heuristic.estimate(node.state)

    # Ex.43
    # Try using A*eps to improve the speed (#dev) with a non-acceptable heuristic.
    # TODO: Create an instance of `AStarEpsilon` with the `MDAMSTAirDistHeuristic`.
    #       Solve the `small_mda_problem_with_distance_cost` with it and print the results.
    #       Use focal_epsilon=0.23, and max_focal_size=40.
    #       Use within_focal_priority_function=within_focal_h_sum_priority_function. This function
    #        (defined just above) is internally using the `MDASumAirDistHeuristic`.
    exit()  # TODO: remove!


def mda_problem_anytime_astar_experiments():
    print()
    print('Solve the MDA problem (moderate input, only distance objective, Anytime-A*, '
          'MSTAirDist heuristics).')

    moderate_mda_problem_with_distance_cost = get_mda_problem('moderate', MDAOptimizationObjective.Distance)

    # Ex.46
    # TODO: create an instance of `AnytimeAStar` once with the `MDAMSTAirDistHeuristic`, with
    #       `max_nr_states_to_expand_per_iteration` set to 1000, solve the
    #       `moderate_mda_problem_with_distance_cost` with it and print the results.
    exit()  # TODO: remove!


def run_all_experiments():
    print('Running all experiments')
    toy_map_problem_experiments()
    basic_mda_problem_experiments()
    mda_problem_with_astar_experiments()
    mda_problem_with_weighted_astar_experiments()
    monetary_cost_objectives_mda_problem_experiments()
    multiple_objectives_mda_problem_experiments()
    mda_problem_with_astar_epsilon_experiments()
    mda_problem_anytime_astar_experiments()


if __name__ == '__main__':
    run_all_experiments()
