from __future__ import annotations

from typing import TYPE_CHECKING

from algorithms.utils import bfs_distance, dijkstra

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
    # Si el juego ya terminó, devolver valores extremos
    if state.is_win():
        return 1000.0
    if state.is_lose():
        return -1000.0

    # Posición actual del dron
    posicion_dron = state.get_drone_position()

    # Si por alguna razón no existe, el estado es terrible
    if posicion_dron is None:
        return -1000.0

    # Información del mapa y entidades
    mapa = state.get_layout()
    entregas_pendientes = state.get_pending_deliveries()
    posiciones_cazadores = state.get_hunter_positions()

    # Valor inicial basado en el score del juego
    valor = 0.45 * state.get_score()

    # Penalización por entregas que aún faltan
    valor -= 220.0 * len(entregas_pendientes)

    # Variables para guardar las mejores métricas encontradas
    costo_entrega_cercana = float("inf")
    pasos_entrega_cercana = float("inf")
    mejor_margen_entrega = float("-inf")
    mejor_bonus_seguridad = 0.0

    # Analizar cada entrega pendiente
    for entrega in entregas_pendientes:

        # Distancia real considerando costos (Dijkstra)
        costo, _ = dijkstra(mapa, posicion_dron, entrega)
        costo_entrega_cercana = min(costo_entrega_cercana, costo)

        # Distancia en pasos simples (BFS)
        pasos_dron = bfs_distance(mapa, posicion_dron, entrega)
        pasos_entrega_cercana = min(pasos_entrega_cercana, pasos_dron)

        # Distancia del cazador más cercano a esa entrega
        pasos_cazador = float("inf")
        for pos in posiciones_cazadores:
            d = bfs_distance(mapa, pos, entrega, hunter_restricted=True)
            if d < pasos_cazador:
                pasos_cazador = d
                
        # Diferencia entre cazador y dron
        margen = pasos_cazador - pasos_dron
        mejor_margen_entrega = max(mejor_margen_entrega, margen)

        # Bonus si el dron puede llegar antes que los cazadores
        if pasos_dron < pasos_cazador:
            bonus = 120.0 / (pasos_dron + 1.0)
            bonus += 20.0 * min(margen, 3)

            mejor_bonus_seguridad = max(mejor_bonus_seguridad, bonus)

    # --- Evaluar distancia a entregas ---

    if costo_entrega_cercana < float("inf"):

        # Preferir entregas cercanas
        valor += 130.0 / (costo_entrega_cercana + 1.0)

        # Penalizar si está lejos
        valor -= 2.0 * costo_entrega_cercana

        # Bonus si está muy cerca
        if costo_entrega_cercana <= 1:
            valor += 55.0
        elif costo_entrega_cercana <= 2:
            valor += 25.0

    elif entregas_pendientes:
        # Si hay entregas pero no se pueden alcanzar
        valor -= 250.0

    # Evaluación usando pasos BFS
    if pasos_entrega_cercana < float("inf"):
        valor += 220.0 / (pasos_entrega_cercana + 1.0)
        valor -= 10.0 * pasos_entrega_cercana

    # --- Evaluar ventaja frente a cazadores ---

    if mejor_margen_entrega > float("-inf"):
        # Si el dron llega antes
        if mejor_margen_entrega > 0:
            valor += 32.0 * min(mejor_margen_entrega, 5)

        # Si los cazadores llegan antes
        else:
            valor += 18.0 * max(mejor_margen_entrega, -4)

    # Bonus adicional por entrega segura
    valor += mejor_bonus_seguridad

    # --- Distancia entre cazadores y dron ---

    distancias_cazadores = []
    for pos in posiciones_cazadores:
        d = bfs_distance(mapa, pos, posicion_dron, hunter_restricted=True)
        distancias_cazadores.append(d)

    cazadores_alcanzables = []
    for d in distancias_cazadores:
        if d < float("inf"):
            cazadores_alcanzables.append(d)

    # Si ningún cazador puede alcanzarlo
    if not cazadores_alcanzables:
        valor += 10.0

    else:

        cazador_mas_cercano = min(cazadores_alcanzables)

        # Penalización fuerte si están demasiado cerca
        if cazador_mas_cercano <= 1:
            valor -= 650.0
        elif cazador_mas_cercano == 2:
            valor -= 180.0
        elif cazador_mas_cercano == 3:
            valor -= 70.0
        elif cazador_mas_cercano == 4:
            valor -= 20.0
        else:
            valor += 10.0 * min(cazador_mas_cercano, 6)
        # Bonus por mantener distancia promedio
        distancia_promedio = sum(cazadores_alcanzables) / len(cazadores_alcanzables)
        valor += 2.0 * min(distancia_promedio, 6)
    # Limitar el valor final al rango permitido
    return max(-1000.0, min(1000.0, valor))
