from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import algorithms.evaluation as evaluation
from algorithms.utils import bfs_distance
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
        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None

# Implementacion basada en el libro de AIMA, página 305, con mejoras hechas por IA para penalizar acciones STOP en nodos MAX y evitar que el dron se quede quieto sin necesidad.
class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for the drone (MAX) vs hunters (MIN) game.
    """

    STOP_PENALTY = 120.0

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using minimax.

        Tips:
        - The game tree alternates: drone (MAX) -> hunter1 (MIN) -> hunter2 (MIN) -> ... -> drone (MAX) -> ...
        - Use self.depth to control the search depth. depth=1 means the drone moves once and each hunter moves once.
        - Use state.get_legal_actions(agent_index) to get legal actions for a specific agent.
        - Use state.generate_successor(agent_index, action) to get the successor state after an action.
        - Use state.is_win() and state.is_lose() to check terminal states.
        - Use state.get_num_agents() to get the total number of agents.
        - Use self.evaluation_function(state) to evaluate leaf/terminal states.
        - The next agent is (agent_index + 1) % num_agents. Depth decreases after all agents have moved (full ply).
        - Return the ACTION (not the value) that maximizes the minimax value for the drone.
        """
        acciones = state.get_legal_actions(self.index)
        if not acciones:
            return None

        mejor_accion = None
        mejor_valor = float("-inf")

        for accion in acciones:
            sucesor = state.generate_successor(self.index, accion)

            valor = self.minimax_value(
                sucesor,
                agente=1,
                profundidad=self.depth
            )
            # Mejora hecha con IA: penalizar acciones STOP en nodos MAX para evitar que el dron se quede quieto sin necesidad.
            if accion == Directions.STOP:
                valor -= self.STOP_PENALTY

            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = accion

        return mejor_accion
    
    def minimax_value(self, estado: GameState, agente: int, profundidad: int) -> float:
            # Caso terminal
        if estado.is_win() or estado.is_lose() or profundidad == 0:
            return self.evaluation_function(estado)

        acciones = estado.get_legal_actions(agente)

        if not acciones:
            return self.evaluation_function(estado)

        num_agentes = estado.get_num_agents()
        siguiente_agente = (agente + 1) % num_agentes
        siguiente_profundidad = profundidad - 1 if siguiente_agente == 0 else profundidad

        # Caso MAX (dron)
        if agente == 0:
            valor = float("-inf")
            for accion in acciones:
                sucesor = estado.generate_successor(agente, accion)
                valor_hijo = self.minimax_value(
                    sucesor,
                    siguiente_agente,
                    siguiente_profundidad
                )
                # Mejora hecha con IA: penalizar acciones STOP en nodos MAX para evitar que el dron se quede quieto sin necesidad.
                if accion == Directions.STOP:
                    valor_hijo -= self.STOP_PENALTY
                valor = max(valor, valor_hijo)
            return valor
        # Caso MIN (cazadores)
        else:
            valor = float("inf")
            for accion in acciones:
                sucesor = estado.generate_successor(agente, accion)
                valor = min(
                    valor,
                    self.minimax_value(
                        sucesor,
                        siguiente_agente,
                        siguiente_profundidad
                    )
                )
            return valor

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta pruning agent. Same as Minimax but with alpha-beta pruning.
    MAX node: prune when value > beta (strict).
    MIN node: prune when value < alpha (strict).
    """

    STOP_PENALTY = 120.0

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
        acciones = state.get_legal_actions(self.index)
        if not acciones:
            return None
        # Se agregan las variables alpha y beta para el algoritmo de poda alfa-beta, y se inicializan con los valores extremos.
        alpha = float("-inf")
        beta = float("inf")
        mejor_accion = None
        mejor_valor = float("-inf")

        for accion in acciones:
            sucesor = state.generate_successor(self.index, accion)
            valor = self.alphabeta_value(
                sucesor,
                agente=1,
                profundidad=self.depth,
                alpha=alpha,
                beta=beta,
            )

            # Evitar que el dron se quede quieto sin necesidad penalizando acciones STOP en nodos MAX.
            if accion == Directions.STOP:
                valor -= self.STOP_PENALTY

            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = accion

            alpha = max(alpha, mejor_valor)

        return mejor_accion

    def alphabeta_value( # Equivalente a minimax_value pero con poda alfa-beta
        self,
        estado: GameState,
        agente: int,
        profundidad: int,
        alpha: float,
        beta: float,
    ) -> float:
        if estado.is_win() or estado.is_lose() or profundidad == 0:
            return self.evaluation_function(estado)
        acciones = estado.get_legal_actions(agente)
        if not acciones:
            return self.evaluation_function(estado)

        num_agentes = estado.get_num_agents()
        siguiente_agente = (agente + 1) % num_agentes
        siguiente_profundidad = profundidad - 1 if siguiente_agente == 0 else profundidad

        # Nodo MAX (dron)
        if agente == 0:
            valor = float("-inf")
            for accion in acciones:
                sucesor = estado.generate_successor(agente, accion)
                valor_hijo = self.alphabeta_value(
                    sucesor,
                    siguiente_agente,
                    siguiente_profundidad,
                    alpha,
                    beta,
                )
                if accion == Directions.STOP:
                    valor_hijo -= self.STOP_PENALTY

                valor = max(valor, valor_hijo)

                # Hacer poda estricta para MAX: podar solo cuando    value > beta
                if valor > beta:
                    return valor
                # Se actualiza alpha después de procesar un nodo MAX, ya que es el valor mínimo que el nodo MIN superior puede garantizar.
                alpha = max(alpha, valor)
            return valor

        # Nodo MIN (cazadores)
        valor = float("inf")
        for accion in acciones:
            sucesor = estado.generate_successor(agente, accion)
            valor = min(
                valor,
                self.alphabeta_value(
                    sucesor,
                    siguiente_agente,
                    siguiente_profundidad,
                    alpha,
                    beta,
                ),
            )
            # Hacer poda estricta para MIN: podar solo cuando value < alpha
            if valor < alpha:
                return valor
            # Se actualiza beta después de procesar un nodo MIN, ya que es el valor máximo que el nodo MAX superior puede garantizar.
            beta = min(beta, valor)
        return valor


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