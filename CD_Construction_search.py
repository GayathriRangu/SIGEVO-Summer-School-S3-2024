#!/usr/bin/env python3
#
# Copyright (C) 2023 Alexandre Jesus <https://adbjesus.com>, Carlos M. Fonseca <cmfonsec@dei.uc.pt>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TextIO, Optional, Any
from collections.abc import Iterable, Hashable
from itertools import combinations


import logging

Objective = Any

@dataclass
class Component:
    node: int
    community_index: Optional[int]

class Solution:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.communities = []
        self.unused = set(range(problem.nnodes))
        self.dist = 0.0

    def output(self) -> str:
        """Output the communities in a readable format, adjusting for one-based indexing."""
        return "\n".join(" ".join(map(lambda x: str(x+1), sorted(community))) for community in self.communities)

    def copy(self):
        # i think that we are not supposed to deepcopy it?
        return self.__class__(self.problem, deepcopy(self.communities), deepcopy(self.unused), self.dist)

    def is_feasible(self) -> bool:
        # There should be no overlapping between the two sets.
        """Check if all nodes are included and there are no overlapping communities."""
        all_nodes = set().union(*self.communities)
        return len(all_nodes) == self.problem.nnodes and not self.unused

    def objective(self) -> Optional[float]:
        # API should be minimize
        # objective fucniton should be updated.
        """Calculate the objective if the solution is feasible."""
        if self.is_feasible():
            return self.dist
        return None

    def add_moves(self) -> Iterable[Component]:
        """Generate all possible moves for each unused node to any existing or new community."""
        if self.unused:
            node_to_add = next(iter(self.unused))  # Take one unused node
            for index in range(len(self.communities)):
                yield Component(node_to_add, index)
            yield Component(node_to_add, None)  # Consider starting a new community

    def add(self, component: Component) -> None:
        """Add a node to a specified community based on the component details."""
        self.add_node_to_community(component.node, component.community_index)

    def lower_bound_incr_add(self, component: Component) -> Optional[float]:
        if component.community_index is not None and component.community_index < len(self.communities):
            total_positive = sum(self.problem.weights[component.node][member] for member in self.communities[component.community_index] if self.problem.weights[component.node][member] > 0)
            total_negative = sum(abs(self.problem.weights[component.node][member]) for member in self.communities[component.community_index] if self.problem.weights[component.node][member] < 0)
            return - (total_positive - total_negative)
        return 0

    def add_node_to_community(self, node: int, community_index: Optional[int] = None):
        if community_index is None or community_index >= len(self.communities):
            self.communities.append({node})
            self.unused.discard(node)
        else:
            positive_weights = sum(self.problem.weights[node][member] for member in self.communities[community_index] if self.problem.weights[node][member] > 0)
            negative_weights = sum(abs(self.problem.weights[node][member]) for member in self.communities[community_index] if self.problem.weights[node][member] < 0)
            
            if positive_weights > negative_weights:
                self.communities[community_index].add(node)
                self.unused.discard(node)
                self.dist += positive_weights - negative_weights
            else:
                self.communities.append({node})
                self.unused.discard(node)

class Problem:
    def __init__(self, nnodes: int, weights: List[List[float]]) -> None:
        """
        Initialize the Problem with number of nodes and the weight matrix.
        """
        self.nnodes = nnodes
        self.weights = weights

    @classmethod
    def from_textio(cls, f: TextIO) -> 'Problem':
        """
        Create a problem from a text I/O source `f`
        """
        n = int(f.readline().strip())  # Read the number of nodes
        weights = [[0.0] * n for _ in range(n)]  # Initialize a square matrix filled with zeros

        for i in range(n):
            line_weights = list(map(float, f.readline().strip().split()))
            # Properly fill weights matrix ensuring symmetry
            for j, weight in enumerate(line_weights):
                if i + j + 1 <= n:  # Check the bounds considering the zero-based index
                    weights[i][i + j] = weight
                    weights[i + j][i] = weight  # Fill symmetrically to denote undirected edges

        print(weights)
        return cls(n, weights)

    def empty_solution(self) -> 'Solution':
        """
        Create an initial solution where there are no communities.
        """
        return Solution(self)



if __name__ == '__main__':
    from api.solvers import *
    from time import perf_counter
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--log-level',
                        choices=['critical', 'error', 'warning', 'info', 'debug'],
                        default='warning')
    parser.add_argument('--log-file', type=argparse.FileType('w'), default=sys.stderr)
    parser.add_argument('--csearch',
                        choices=['beam', 'grasp', 'greedy', 'heuristic', 'as', 'mmas', 'none'],
                        default='none')
    parser.add_argument('--cbudget', type=float, default=5.0)
    parser.add_argument('--lsearch',
                        choices=['bi', 'fi', 'ils', 'rls', 'sa', 'none'],
                        default='none')
    parser.add_argument('--lbudget', type=float, default=5.0)
    parser.add_argument('--input-file', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('--output-file', type=argparse.FileType('w'), default=sys.stdout)
    args = parser.parse_args()

    logging.basicConfig(stream=args.log_file,
                        level=args.log_level.upper(),
                        format="%(levelname)s;%(asctime)s;%(message)s")

    p = Problem.from_textio(args.input_file)
    s: Optional[Solution] = p.empty_solution()

    start = perf_counter()

    if s is not None:
        if args.csearch == 'heuristic':
            s = heuristic_construction(s)
        elif args.csearch == 'greedy':
            s = greedy_construction(s)
        elif args.csearch == 'beam':
            s = beam_search(s, 10)
        elif args.csearch == 'grasp':
            s = grasp(s, args.cbudget, alpha = 0.01)
        elif args.csearch == 'as':
            ants = [s]*100
            s = ant_system(ants, args.cbudget, beta = 5.0, rho = 0.5, tau0 = 1 / 3000.0)
        elif args.csearch == 'mmas':
            ants = [s]*100
            s = mmas(ants, args.cbudget, beta = 5.0, rho = 0.02, taumax = 1 / 3000.0, globalratio = 0.5)

    if s is not None:
        if args.lsearch == 'bi':
            s = best_improvement(s, args.lbudget)
        elif args.lsearch == 'fi':
            s = first_improvement(s, args.lbudget) 
        elif args.lsearch == 'ils':
            s = ils(s, args.lbudget)
        elif args.lsearch == 'rls':
            s = rls(s, args.lbudget)
        elif args.lsearch == 'sa':
            s = sa(s, args.lbudget, 30)

    end = perf_counter()

    if s is not None:
        print(s.output(), file=args.output_file)
        if s.objective() is not None:
            logging.info(f"Objective: {s.objective():.3f}")
        else:
            logging.info(f"Objective: None")
    else:
        logging.info(f"Objective: no solution found")

    logging.info(f"Elapsed solving time: {end-start:.4f}")

