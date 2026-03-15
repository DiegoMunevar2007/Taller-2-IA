from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import algorithms.evaluation as evaluation
from world.game import Agent, Directions

if TYPE_CHECKING:
    from world.game_state import GameState


class MultiAgentSearchAgent(Agent, ABC):
    """
    Base class for multi-agent search agents (Minimax, AlphaBeta, Expectimax).
    """

    def __init__(self, depth: str = "2", _index: int = 0, prob: str = "0.0") -> None:
        self.index = 0  # Drone is always agent 0
        self.depth = int(depth)
        self.prob = float(
            prob
        )  # Probability that each hunter acts randomly (0=greedy, 1=random)
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone from the current GameState.
        """
        pass


class RandomAgent(MultiAgentSearchAgent):
    """
    Agent that chooses a legal action uniformly at random.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Get a random legal action for the drone.
        """
        acciones_valid = state.get_acciones_valid(self.index)
        return random.choice(acciones_valid) if acciones_valid else None


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for the drone (MAX) vs hunters (MIN) game.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using minimax.

        Tips:
        - The game tree alternates: drone (MAX) -> hunter1 (MIN) -> hunter2 (MIN) -> ... -> drone (MAX) -> ...
        - Use self.depth to control the search depth. depth=1 means the drone moves once and each hunter moves once.
        - Use state.get_acciones_valid(agent_index) to get legal actions for a specific agent.
        - Use state.generate_successor(agent_index, action) to get the successor state after an action.
        - Use state.is_win() and state.is_lose() to check terminal states.
        - Use state.get_numero_agentes() to get the total number of agents.
        - Use self.evaluation_function(state) to evaluate leaf/terminal states.
        - The next agent is (agent_index + 1) % numero_agentes. Depth decreases after all agents have moved (full ply).
        - Return the ACTION (not the value) that maximizes the minimax value for the drone.
        """
        numero_agentes = state.get_numero_agentes()

        def minimax(current_state: GameState, agent_index: int, depth: int) -> float:
            # Caso base: estado terminal o profundidad agotada
            if current_state.is_win() or current_state.is_lose() or depth == 0:
                return self.evaluation_function(current_state)

            acciones_valid = current_state.get_acciones_valid(agent_index)
            if not acciones_valid:
                return self.evaluation_function(current_state)

            if agent_index == 0 and Directions.STOP in acciones_valid and len(acciones_valid) > 1:
                acciones_valid = [a for a in acciones_valid if a != Directions.STOP]

            next_agent = (agent_index + 1) % numero_agentes
            next_depth = depth - 1 if next_agent == 0 else depth

            # MAX: dron
            if agent_index == 0:
                value = float("-inf")
                for action in acciones_valid:
                    successor = current_state.generate_successor(agent_index, action)
                    value = max(value, minimax(successor, next_agent, next_depth))
                return value

            # MIN: cazadores
            else:
                value = float("inf")
                for action in acciones_valid:
                    successor = current_state.generate_successor(agent_index, action)
                    value = min(value, minimax(successor, next_agent, next_depth))
                return value


        acciones_valid = state.get_acciones_valid(0)
        if not acciones_valid:
            return None

        if Directions.STOP in acciones_valid and len(acciones_valid) > 1:
            acciones_valid = [a for a in acciones_valid if a != Directions.STOP]

        best_value = float("-inf")
        best_action = None

        for action in acciones_valid:
            successor = state.generate_successor(0, action)
            value = minimax(successor, 1 % numero_agentes, self.depth)

            if value > best_value or best_action is None:
                best_value = value
                best_action = action

        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta pruning agent. Same as Minimax but with alpha-beta pruning.
    MAX node: prune when value > beta (strict).
    MIN node: prune when value < alpha (strict).
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using alpha-beta pruning.

        Tips:
        - Same structure as MinimaxAgent, but with alpha-beta pruning.
        - Alpha: best value MAX can guarantee (initially -inf).
        - Beta: best value MIN can guarantee (initially +inf).
        - MAX node: prune when value > beta (strict inequality, do NOT prune on equality).
        - MIN node: prune when value < alpha (strict inequality, do NOT prune on equality).
        - Update alpha at MAX nodes: alpha = max(alpha, value).
        - Update beta at MIN nodes: beta = min(beta, value).
        - Pass alpha and beta through the recursive calls.
        """
        numero_agentes = state.get_numero_agentes()

        def alphabeta(
            current_state: GameState,
            agent_index: int,
            depth: int,
            alpha: float,
            beta: float,
        ) -> float:
            if current_state.is_win() or current_state.is_lose() or depth == 0:
                return self.evaluation_function(current_state)

            acciones_valid = current_state.get_acciones_valid(agent_index)
            if not acciones_valid:
                return self.evaluation_function(current_state)

            if agent_index == 0 and Directions.STOP in acciones_valid and len(acciones_valid) > 1:
                acciones_valid = [a for a in acciones_valid if a != Directions.STOP]

            next_agent = (agent_index + 1) % numero_agentes
            next_depth = depth - 1 if next_agent == 0 else depth

            # MAX: drone
            if agent_index == 0:
                value = float("-inf")
                for action in acciones_valid:
                    successor = current_state.generate_successor(agent_index, action)
                    value = max(
                        value,
                        alphabeta(successor, next_agent, next_depth, alpha, beta)
                    )

                    if value > beta:   # strict pruning
                        return value

                    alpha = max(alpha, value)

                return value

            # MIN: hunters
            else:
                value = float("inf")
                for action in acciones_valid:
                    successor = current_state.generate_successor(agent_index, action)
                    value = min(
                        value,
                        alphabeta(successor, next_agent, next_depth, alpha, beta)
                    )

                    if value < alpha:   # strict pruning
                        return value

                    beta = min(beta, value)

                return value

        acciones_valid = state.get_acciones_valid(0)
        if not acciones_valid:
            return None

        if Directions.STOP in acciones_valid and len(acciones_valid) > 1:
            acciones_valid = [a for a in acciones_valid if a != Directions.STOP]

        best_value = float("-inf")
        best_action = None
        alpha = float("-inf")
        beta = float("inf")

        for action in acciones_valid:
            successor = state.generate_successor(0, action)
            value = alphabeta(successor, 1 % numero_agentes, self.depth, alpha, beta)

            if value > best_value or best_action is None:
                best_value = value
                best_action = action

            alpha = max(alpha, best_value)

        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent with a mixed hunter model.

    Each hunter acts randomly with probability self.prob and greedily
    (worst-case / MIN) with probability 1 - self.prob.

    * When prob = 0:  behaves like Minimax (hunters always play optimally).
    * When prob = 1:  pure expectimax (hunters always play uniformly at random).
    * When 0 < prob < 1: weighted combination that correctly models the
      actual MixedHunterAgent used at game-play time.

    Chance node formula:
        value = (1 - p) * min(child_values) + p * mean(child_values)
    """

    def expectimax(self, state: GameState, agent_index: int, depth: int) -> float:
        if state.is_win() or state.is_lose() or depth == 0:
            return self.evaluation_function(state)

        acciones_valid = state.get_legal_actions(agent_index)
        if not acciones_valid:
            return self.evaluation_function(state)

        if agent_index == 0 and Directions.STOP in acciones_valid and len(acciones_valid) > 1:
            filtrado = []
            for a in acciones_valid:
                if a != Directions.STOP:
                    filtrado.append(a)
            acciones_valid = filtrado

        num_agents = state.get_num_agents()
        next_agent = (agent_index + 1) % num_agents
        next_depth = depth - 1 if next_agent == 0 else depth

        if agent_index == 0:
            return self._max_value(state, agent_index, acciones_valid, next_agent, next_depth)
        else:
            return self._chance_value(state, agent_index, acciones_valid, next_agent, next_depth)

    def _max_value( self, state: GameState, agent_index: int, acciones_valid: list, next_agent: int, next_depth: int) -> float:
        value = float("-inf")
        for accion in acciones_valid:
            sig_state = state.generate_successor(agent_index, accion)
            value = max(value, self.expectimax(sig_state, next_agent, next_depth))
        return value

    def _chance_value( self, state: GameState, agent_index: int, acciones_valid: list, next_agent: int, next_depth: int) -> float:
        child_values = []
        for accion in acciones_valid:
            sig_state = state.generate_successor(agent_index, accion)
            child_values.append(self.expectimax(sig_state, next_agent, next_depth))
        greedy_value = min(child_values)
        random_value = sum(child_values) / len(child_values)
        return (1 - self.prob) * greedy_value + self.prob * random_value

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using expectimax with mixed hunter model.

        Tips:
        - Drone nodes are MAX (same as Minimax).
        - Hunter nodes are CHANCE with mixed model: the hunter acts greedily with
          probability (1 - self.prob) and uniformly at random with probability self.prob.
        - Mixed expected value = (1-p) * min(child_values) + p * mean(child_values).
        - When p=0 this reduces to Minimax; when p=1 it is pure uniform expectimax.
        - Do NOT prune in expectimax (unlike alpha-beta).
        - self.prob is set via the constructor argument prob.
        """
        num_agents = state.get_num_agents()
        acciones_valid = state.get_legal_actions(0)
        if not acciones_valid:
            return None

        if Directions.STOP in acciones_valid and len(acciones_valid) > 1:
            filtrado = []
            for accion in acciones_valid:
                if accion != Directions.STOP:
                    filtrado.append(accion)
            acciones_valid = filtrado

        best_value = float("-inf")
        best_action = None

        for accion in acciones_valid:
            sig_state = state.generate_successor(0, accion)
            value = self.expectimax(sig_state, 1 % num_agents, self.depth)

            if value > best_value or best_action is None:
                best_value = value
                best_action = accion

        return best_action
