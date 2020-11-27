import math
from typing import *
from dataclasses import dataclass
from enum import Enum

from framework import *
from .map_heuristics import AirDistHeuristic
from .cached_map_distance_finder import CachedMapDistanceFinder
from .mda_problem_input import *

__all__ = ['MDAState', 'MDACost', 'MDAProblem', 'MDAOptimizationObjective']


@dataclass(frozen=True)
class MDAState(GraphProblemState):
    """
    An instance of this class represents a state of MDA problem.
    This state includes:
        `current_site`:
            The current site where the ambulate is at.
            The initial state stored in this field the initial ambulance location (which is a `Junction` object).
            Other states stores the last visited reported apartment (object of type `ApartmentWithSymptomsReport`),
             or the last visited laboratory (object of type `Laboratory`).
        `tests_on_ambulance`:
            Stores the reported-apartments (objects of type `ApartmentWithSymptomsReport`) which had been visited,
             and their tests are still stored on the ambulance (hasn't been transferred to a laboratory yet).
        `tests_transferred_to_lab`:
            Stores the reported-apartments (objects of type `ApartmentWithSymptomsReport`) which had been visited,
             and their tests had already been transferred to a laboratory.
        `nr_matoshim_on_ambulance`:
            The number of matoshim currently stored on the ambulance.
            Whenever visiting a reported apartment, this number is decreased by the #roommates in this apartment.
            Whenever visiting a laboratory for the first time, we transfer the available matoshim from this lab
             to the ambulance.
        `visited_labs`:
            Stores the laboratories (objects of type `Laboratory`) that had been visited at least once.
    """

    current_site: Union[Junction, Laboratory, ApartmentWithSymptomsReport]
    tests_on_ambulance: FrozenSet[ApartmentWithSymptomsReport]
    tests_transferred_to_lab: FrozenSet[ApartmentWithSymptomsReport]
    nr_matoshim_on_ambulance: int
    visited_labs: FrozenSet[Laboratory]

    @property
    def current_location(self):
        if isinstance(self.current_site, ApartmentWithSymptomsReport) or isinstance(self.current_site, Laboratory):
            return self.current_site.location
        assert isinstance(self.current_site, Junction)
        return self.current_site

    def get_current_location_short_description(self) -> str:
        if isinstance(self.current_site, ApartmentWithSymptomsReport):
            return f'test @ {self.current_site.reporter_name}'
        if isinstance(self.current_site, Laboratory):
            return f'lab {self.current_site.name}'
        return 'initial-location'

    def __str__(self):
        return f'(' \
               f'loc: {self.get_current_location_short_description()} ' \
               f'tests on ambulance: ' \
               f'{[f"{reported_apartment.reporter_name} ({reported_apartment.nr_roommates})" for reported_apartment in self.tests_on_ambulance]} ' \
               f'tests transferred to lab: ' \
               f'{[f"{reported_apartment.reporter_name} ({reported_apartment.nr_roommates})" for reported_apartment in self.tests_transferred_to_lab]} ' \
               f'#matoshim: {self.nr_matoshim_on_ambulance} ' \
               f'visited labs: {[lab.name for lab in self.visited_labs]}' \
               f')'

    def __eq__(self, other):
        """
        This method is used to determine whether two given state objects represent the same state.
        """
        assert isinstance(other, MDAState)

        # TODO [Ex.17]: Complete the implementation of this method!
        #  Note that you can simply compare two instances of `Junction` type
        #   (using equals `==` operator) because the class `Junction` explicitly
        #   implements the `__eq__()` method. The types `frozenset`, `ApartmentWithSymptomsReport`, `Laboratory`
        #   are also comparable (in the same manner).

        return self.current_site == other.current_site and \
               self.tests_on_ambulance == other.tests_on_ambulance and \
               self.tests_transferred_to_lab == other.tests_transferred_to_lab and \
               self.nr_matoshim_on_ambulance == other.nr_matoshim_on_ambulance and \
               self.visited_labs == other.visited_labs

    def __hash__(self):
        """
        This method is used to create a hash of a state instance.
        The hash of a state being is used whenever the state is stored as a key in a dictionary
         or as an item in a set.
        It is critical that two objects representing the same state would have the same hash!
        """
        return hash((self.current_site, self.tests_on_ambulance, self.tests_transferred_to_lab,
                     self.nr_matoshim_on_ambulance, self.visited_labs))

    def get_total_nr_tests_taken_and_stored_on_ambulance(self) -> int:
        """
        This method returns the total number of of tests that are stored on the ambulance in this state.
        TODO [Ex.17]: Implement this method.
         Notice that this method can be implemented using a single line of code - do so!
         Use python's built-it `sum()` function.
         Notice that `sum()` can receive an *ITERATOR* as argument; That is, you can simply write something like this:
        >>> sum(<some expression using item> for item in some_collection_of_items)
        """
        return sum(item.nr_roommates for item in self.tests_on_ambulance)


class MDAOptimizationObjective(Enum):
    Distance = 'Distance'
    Monetary = 'Monetary'
    TestsTravelDistance = 'TestsTravelDistance'


@dataclass(frozen=True)
class MDACost(ExtendedCost):
    """
    An instance of this class is returned as an operator cost by the method
     `MDAProblem.expand_state_with_costs()`.
    The `SearchNode`s that will be created during the run of the search algorithm are going
     to have instances of `MDACost` in SearchNode's `cost` field (instead of float values).
    The reason for using a custom type for the cost (instead of just using a `float` scalar),
     is because we want the cumulative cost (of each search node and particularly of the final
     node of the solution) to be consisted of 3 objectives:
     (i) distance, (ii) money, and (iii) tests-travel.
    The field `optimization_objective` controls the objective of the problem (the cost we want
     the solver to minimize). In order to tell the solver which is the objective to optimize,
     we have the `get_g_cost()` method, which returns a single `float` scalar which is only the
     cost to optimize.
    This way, whenever we get a solution, we can inspect the 2 different costs of that solution,
     even though the objective was only one of the costs.
    Having said that, note that during this assignment we will mostly use the distance objective.
    """
    distance_cost: float = 0.0
    monetary_cost: float = 0.0
    tests_travel_distance_cost: float = 0.0
    optimization_objective: MDAOptimizationObjective = MDAOptimizationObjective.Monetary

    def __add__(self, other):
        assert isinstance(other, MDACost)
        assert other.optimization_objective == self.optimization_objective
        return MDACost(
            optimization_objective=self.optimization_objective,
            distance_cost=self.distance_cost + other.distance_cost,
            monetary_cost=self.monetary_cost + other.monetary_cost,
            tests_travel_distance_cost=self.tests_travel_distance_cost + other.tests_travel_distance_cost)

    def get_g_cost(self) -> float:
        if self.optimization_objective == MDAOptimizationObjective.Distance:
            return self.distance_cost
        elif self.optimization_objective == MDAOptimizationObjective.Monetary:
            return self.monetary_cost
        assert self.optimization_objective == MDAOptimizationObjective.TestsTravelDistance
        return self.tests_travel_distance_cost

    def __repr__(self):
        return f'MDACost(' \
               f'dist={self.distance_cost:11.3f}m, ' \
               f'money={self.monetary_cost:11.3f}NIS, ' \
               f'tests-travel={self.tests_travel_distance_cost:11.3f}m)'


class MDAProblem(GraphProblem):
    """
    An instance of this class represents an MDA problem.
    """

    name = 'MDA'

    def __init__(self,
                 problem_input: MDAProblemInput,
                 streets_map: StreetsMap,
                 optimization_objective: MDAOptimizationObjective = MDAOptimizationObjective.Distance):
        self.name += f'({problem_input.input_name}({len(problem_input.reported_apartments)}):{optimization_objective.name})'
        initial_state = MDAState(
            current_site=problem_input.ambulance.initial_location,
            tests_on_ambulance=frozenset(),
            tests_transferred_to_lab=frozenset(),
            nr_matoshim_on_ambulance=problem_input.ambulance.initial_nr_matoshim,
            visited_labs=frozenset())
        super(MDAProblem, self).__init__(initial_state)
        self.problem_input = problem_input
        self.streets_map = streets_map
        self.map_distance_finder = CachedMapDistanceFinder(
            streets_map, AStar(AirDistHeuristic))
        self.optimization_objective = optimization_objective

    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[OperatorResult]:
        """
        TODO [Ex.17]: Implement this method!
        This method represents the `Succ: S -> P(S)` function of the MDA problem.
        The `Succ` function is defined by the problem operators as shown in class.
        The MDA problem operators are defined in the assignment instructions.
        It receives a state and iterates over its successor states.
        Notice that this its return type is an *Iterator*. It means that this function is not
         a regular function, but a `generator function`. Hence, it should be implemented using
         the `yield` statement.
        For each successor, an object of type `OperatorResult` is yielded. This object describes the
            successor state, the cost of the applied operator and its name. Look for its definition
            and use the correct fields in its c'tor. The operator name should be in the following
            format: `visit ReporterName` (with the correct reporter name) if an reported-apartment
            visit operator was applied (to take tests from the roommates of an apartment), or
            `go to lab LabName` if a laboratory visit operator was applied.
            The apartment-report object stores its reporter-name in one of its fields.
        Things you might want to use:
            - The method `self.get_total_nr_tests_taken_and_stored_on_ambulance()`.
            - The field `self.problem_input.laboratories`.
            - The field `self.problem_input.ambulance.total_fridges_capacity`.
            - The method `self.get_reported_apartments_waiting_to_visit()` here.
            - The method `self.get_operator_cost()`.
            - The c'tor for `AmbulanceState` to create the new successor state.
            - Python's built-in method `frozenset()` to create a new frozen set (for fields that
              expect this type) from another collection (set/list/tuple/iterator).
            - Other fields of the state and the problem input.
            - Python's sets union operation (`some_set_or_frozenset | some_other_set_or_frozenset`).
        """

        assert isinstance(state_to_expand, MDAState)
        for outgoing_link in state_to_expand.current_site.outgoing_links:
            next_place = self.streets_map.get(outgoing_link.target)
            is_laboratory = isinstance(next_place, Laboratory)
            if is_laboratory:
                name = 'go to lab ' + self.problem_input.laboratories[next_place.index]
                new_tests_on_ambulance = frozenset()
            elif isinstance(next_place, ApartmentWithSymptomsReport):
                name = 'visit ' + next_place.reporter_name
                new_tests_on_ambulance = frozenset({next_place}).union(state_to_expand.tests_on_ambulance)
            else:
                name = None
                new_tests_on_ambulance = state_to_expand.tests_on_ambulance
            new_visited_labs = state_to_expand.visited_labs if not is_laboratory else \
                                                    frozenset({next_place}).union(state_to_expand.visited_labs)
            new_tests_transfer_to_lab = state_to_expand.tests_transferred_to_lab if not is_laboratory \
                else frozenset(state_to_expand.tests_transferred_to_lab.union(state_to_expand.tests_on_ambulance))
            new_nr_matoshim = state_to_expand.nr_matoshim_on_ambulance if not is_laboratory \
                else state_to_expand.nr_matoshim_on_ambulance + next_place.max_nr_matoshim
            new_state = MDAState(next_place, new_tests_on_ambulance, new_tests_transfer_to_lab, new_nr_matoshim, new_visited_labs)

            yield OperatorResult(new_state, self.get_operator_cost(state_to_expand, new_state), name)

    def get_operator_cost(self, prev_state: MDAState, succ_state: MDAState) -> MDACost:
        """
        Calculates the operator cost (of type `MDACost`) of an operator (moving from the `prev_state`
         to the `succ_state`). The `MDACost` type is defined above in this file (with explanations).
        Use the formal MDA problem's operator costs definition presented in the assignment-instructions.
        TODO [Ex.17]: implement this method!
        Use the method `self.map_distance_finder.get_map_cost_between()` to calculate the distance
         between two junctions. This distance is used for calculating the 3 costs.
        If the location of the next state is not reachable (on the streets-map) from the location of
         the previous state, use the value of `float('inf')` for all costs.
        You might want to use the method `MDAState::get_total_nr_tests_taken_and_stored_on_ambulance()`
         both for the tests-travel and the monetary costs.
        For the monetary cost you might want to use the following fields:
         `self.problem_input.ambulance.drive_gas_consumption_liter_per_meter`
         `self.problem_input.gas_liter_price`
         `self.problem_input.ambulance.fridges_gas_consumption_liter_per_meter`
         `self.problem_input.ambulance.fridge_capacity`
         `MDAState::get_total_nr_tests_taken_and_stored_on_ambulance()`
        For calculating the #active-fridges (the monetary cost) you might want to use the
         function `math.ceil(some_float_value)`.
        Note: For calculating sum of a collection (list/tuple/set) in python, you can simply
         use `sum(some_collection)`.
        Note: For getting a slice of an tuple/list in python you can use slicing indexing. examples:
            `some_tuple[:k]` - would create a new tuple with the first `k` elements of `some_tuple`.
            `some_tuple[k:]` - would create a new tuple that is based on `some_tuple` but without
                               its first `k` items.
            `some_tuple[k:n]` - would create a new tuple that is based on `some_tuple` but without
                                its first `k` items and until the `n`-th item.
            You might find this tip useful for summing a slice of a collection.
        """

        # Indicators
        is_lab_indicator = 1 if isinstance(succ_state.current_site, Laboratory) else 0
        is_visited_lab_indicator = 1 if is_lab_indicator == 1 and succ_state.visited_labs.__contains__(
            succ_state.current_site.location) else 0
        is_taken_not_empty_indicator = 1 if prev_state.get_total_nr_tests_taken_and_stored_on_ambulance() != 0 else 0

        # helpers costs
        lab_test_transfer_cost = is_lab_indicator * succ_state.current_site.tests_transfer_cost
        lab_revisit_cost = is_lab_indicator * succ_state.current_site.revisit_extra_cost
        active_fridges = math.ceil(
           succ_state.get_total_nr_tests_taken_and_stored_on_ambulance() / self.problem_input.ambulance.fridge_capacity)
        fridges_gas_consumption_per_meter = sum(
            self.problem_input.ambulance.fridges_gas_consumption_liter_per_meter[:active_fridges])

        # costs
        distance_cost = self.map_distance_finder.get_map_cost_between(prev_state.current_site, succ_state.current_site)
        if distance_cost is None:
            return MDACost(float('inf'), float('inf'), float('inf'))
        monetary_cost = self.problem_input.gas_liter_price * \
                        (self.problem_input.ambulance.drive_gas_consumption_liter_per_meter +
                         fridges_gas_consumption_per_meter) * distance_cost + \
                        is_lab_indicator * (is_taken_not_empty_indicator * lab_test_transfer_cost +
                                            is_visited_lab_indicator * lab_revisit_cost)
        test_travel_cost = distance_cost * prev_state.get_total_nr_tests_taken_and_stored_on_ambulance()
        return MDACost(distance_cost, monetary_cost, test_travel_cost)

    def is_goal(self, state: GraphProblemState) -> bool:
        """
        This method receives a state and returns whether this state is a goal.
        TODO [Ex.17]: implement this method using a single `return` line!
         Use sets/frozensets comparison (`some_set == some_other_set`).
         In order to create a set from some other collection (list/tuple) you can just `set(some_other_collection)`.
        """
        assert isinstance(state, MDAState)
        return isinstance(state.current_location, Laboratory) and not state.tests_on_ambulance and \
               state.tests_transferred_to_lab == set(self.problem_input.reported_apartments)

    def get_zero_cost(self) -> Cost:
        """
        Overridden method of base class `GraphProblem`. For more information, read
         documentation in the default implementation of this method there.
        In this problem the accumulated cost is not a single float scalar, but an
         extended cost, which actually includes 2 scalar costs.
        """
        return MDACost(optimization_objective=self.optimization_objective)

    def get_reported_apartments_waiting_to_visit(self, state: MDAState) -> List[ApartmentWithSymptomsReport]:
        """
        This method returns a list of all reported-apartments that haven't been visited yet.
        For the sake of determinism considerations, the returned list has to be sorted by
         the apartment's report id in an ascending order.
        TODO [Ex.17]: Implement this method.
            Use sets difference operation (`some_set - some_other_set`).
            Use `list(some_set)` to create a list from some given set, and then use
                `some_list_instance.sort(key=...)` to sort this list. Use a `lambda`
                function for the sorting `key` parameter. You can read about it and
                see examples in the internet.
            Note: Given a collection of items, you can create a new set of these items simply by
                `set(my_collection_of_items)`. Then you can use set operations over this newly
                generated set.
            Note: This method can be implemented using a single line of code. Try to do so.
        """

        # initial = set(self.problem_input.reported_apartments)
        # to_remove1 = set(state.tests_transferred_to_lab)
        # to_remove2 = set(state.tests_on_ambulance)
        # to_list = list(initial - to_remove1 - to_remove2)
        # to_list.sort(key=lambda a: ApartmentWithSymptomsReport.report_id)
        # return to_list
        waiting_to_visit = list(set(self.problem_input.reported_apartments) - set(state.tests_transferred_to_lab) -
                       set(state.tests_on_ambulance))
        waiting_to_visit.sort(key=lambda a: ApartmentWithSymptomsReport.report_id)
        return waiting_to_visit
    def get_all_certain_junctions_in_remaining_ambulance_path(self, state: MDAState) -> List[Junction]:
        """
        This method returns a list of junctions that are part of the remaining route of the ambulance.
        This includes the ambulance's current location, and the locations of the reported apartments
         that hasn't been visited yet.
        The list should be ordered by the junctions index ascendingly (small to big).
        TODO [Ex.21]: Implement this method.
            Use the method `self.get_reported_apartments_waiting_to_visit(state)`.
            Use python's `sorted(some_list, key=...)` function.
        """
        raise NotImplementedError  # TODO: remove this line!
