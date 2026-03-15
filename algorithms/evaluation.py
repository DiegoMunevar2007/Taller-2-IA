from __future__ import annotations
from typing import TYPE_CHECKING
from algorithms.utils import dijkstra, bfs_distance

if TYPE_CHECKING:
    from world.game_state import GameState


def evaluation_function(state: GameState) -> float:
    """
    Evaluation function for non-terminal states of the drone vs. hunters game.

    A good evaluation function can consider multiple factors, such as:
      (a) BFS distance from drone to nearest delivery point (closer is better).
          Uses actual path distance so walls and terrain are respected.
      (b) BFS distance from each hunter to the drone, traversing only normal
          terrain ('.' / ' ').  Hunters blocked by mountains, fog, or storms
          are treated as unreachable (distance = inf) and pose no threat.
      (c) BFS distance to a "safe" position (i.e., a position that is not in the path of any hunter).
      (d) Number of pending deliveries (fewer is better).
      (e) Current score (higher is better).
      (f) Delivery urgency: reward the drone for being close to a delivery it can
          reach strictly before any hunter, so it commits to nearby pickups
          rather than oscillating in place out of excessive hunter fear.
      (g) Adding a revisit penalty can help prevent the drone from getting stuck in cycles.

    Returns a value in [-1000, +1000].

    Tips:
    - Use state.get_drone_position() to get the drone's current (x, y) position.
    - Use state.get_hunter_positions() to get the list of hunter (x, y) positions.
    - Use state.get_pending_deliveries() to get the set of pending delivery (x, y) positions.
    - Use state.get_score() to get the current game score.
    - Use state.get_layout() to get the current layout.
    - Use state.is_win() and state.is_lose() to check terminal states.
    - Use bfs_distance(layout, start, goal, hunter_restricted) from algorithms.utils
      for cached BFS distances. hunter_restricted=True for hunter-only terrain.
    - Use dijkstra(layout, start, goal) from algorithms.utils for cached
      terrain-weighted shortest paths, returning (cost, path).
    - Consider edge cases: no pending deliveries, no hunters nearby.
    - A good evaluation function balances delivery progress with hunter avoidance.
    """
    if state.is_win():
        return 1000.0
    if state.is_lose():
        return -1000.0

    drone_pos = state.get_drone_position()
    hunters = state.get_hunter_positions()
    pending = state.get_pending_deliveries()
    layout = state.get_layout()

    value = float(state.get_score())

    # Menos entregas pendientes = mejor
    value -= 120.0 * len(pending)

    # Distancia a la entrega más cercana
    if pending:
        nearest_delivery = float("inf")
        for delivery in pending:
            cost, _ = dijkstra(layout, drone_pos, delivery)
            if cost < nearest_delivery:
                nearest_delivery = cost

        if nearest_delivery != float("inf"):
            value -= 5.0 * nearest_delivery
            value += 80.0 / (1.0 + nearest_delivery)
        else:
            value -= 200.0

    # Distancia del cazador más cercano al dron
    nearest_hunter = float("inf")
    for hunter_pos in hunters:
        dist = bfs_distance(layout, hunter_pos, drone_pos, hunter_restricted=True)
        if dist < nearest_hunter:
            nearest_hunter = dist

    if nearest_hunter != float("inf"):
        value += 15.0 * min(nearest_hunter, 6)

        if nearest_hunter == 0:
            value -= 900.0
        elif nearest_hunter == 1:
            value -= 400.0
        elif nearest_hunter == 2:
            value -= 180.0
    else:
        value += 100.0

    return max(-1000.0, min(1000.0, value))