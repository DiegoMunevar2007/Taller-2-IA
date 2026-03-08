from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algorithms.problems_csp import DroneAssignmentCSP

def backtracking_basico(csp: DroneAssignmentCSP, asignados: dict[str, str]) -> dict[str, str] | None:
    if csp.is_complete(asignados):
        return asignados
      
    variable_sin_asignar = csp.get_unassigned_variables(asignados)[0] # Obtener las variables no asignadas
    
    for valor in csp.domains[variable_sin_asignar]:
        if csp.is_consistent(variable_sin_asignar, valor, asignados): # Verificar si el valor es consistente con las restricciones 
            csp.assign(variable_sin_asignar, valor, asignados)
            resultado = backtracking_basico(csp, asignados)
            if resultado is not None:
                return resultado
            csp.unassign(variable_sin_asignar, asignados)
    return None
    
    
def eliminar_inconsistencias_fw(csp: DroneAssignmentCSP, vecinos: list[str], asignados: dict[str, str]):
    eliminados = []

    for vecino in vecinos: # Se miran todos los vecinos no asignados, y se eliminan los valores inconsistentes con la asignación actual
        if vecino not in asignados:
            for val_vecino in csp.domains[vecino]:
                prueba = asignados.copy() # Se mantiene una copia de la asignación actual para probar cada valor del vecino
                prueba[vecino] = val_vecino

                if not csp.is_consistent(vecino, val_vecino, prueba):
                    csp.domains[vecino].remove(val_vecino)
                    eliminados.append((vecino, val_vecino))

    return eliminados


def restaurar_dominios(csp: DroneAssignmentCSP, eliminados):
    for var, val in eliminados:
        csp.domains[var].append(val)


def backtracking_forward_checking(csp: DroneAssignmentCSP, asignados: dict[str, str]) -> dict[str, str] | None:
    if csp.is_complete(asignados):
        return asignados

    variable = csp.get_unassigned_variables(asignados)[0]

    for valor in csp.domains[variable]:
      if csp.is_consistent(variable, valor, asignados):
            csp.assign(variable, valor, asignados)

            vecinos = csp.get_neighbors(variable)
            eliminados = eliminar_inconsistencias_fw(csp, vecinos, asignados)

            resultado = backtracking_forward_checking(csp, asignados)
            if resultado is not None:
                return resultado

            restaurar_dominios(csp, eliminados)
            csp.unassign(variable, asignados)

    return None


def backtracking_search(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Basic backtracking search without optimizations.

    Tips:
    - An assignment is a dictionary mapping variables to values (e.g. {X1: Cell(1,2), X2: Cell(3,4)}).
    - Use csp.assign(var, value, assignment) to assign a value to a variable.
    - Use csp.unassign(var, assignment) to unassign a variable.
    - Use csp.is_consistent(var, value, assignment) to check if an assignment is consistent with the constraints.
    - Use csp.is_complete(assignment) to check if the assignment is complete (all variables assigned).
    - Use csp.get_unassigned_variables(assignment) to get a list of unassigned variables.
    - Use csp.domains[var] to get the list of possible values for a variable.
    - Use csp.get_neighbors(var) to get the list of variables that share a constraint with var.
    - Add logs to measure how good your implementation is (e.g. number of assignments, backtracks).

    You can find inspiration in the textbook's pseudocode:
    Artificial Intelligence: A Modern Approach (4th Edition) by Russell and Norvig, Chapter 5: Constraint Satisfaction Problems
    """
    return backtracking_basico(csp, {}) # Se usa backtracking_search como una función máscara para llamar a la función recursiva


def backtracking_fc(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with Forward Checking.

    Tips:
    - Forward checking: After assigning a value to a variable, eliminate inconsistent values from
      the domains of unassigned neighbors. If any neighbor's domain becomes empty, backtrack immediately.
    - Save domains before forward checking so you can restore them on backtrack.
    - Use csp.get_neighbors(var) to get variables that share constraints with var.
    - Use csp.is_consistent(neighbor, val, assignment) to check if a value is still consistent.
    - Forward checking reduces the search space by detecting failures earlier than basic backtracking.
    """
    return backtracking_forward_checking(csp, {})


def backtracking_ac3(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with AC-3 arc consistency.

    Tips:
    - AC-3 enforces arc consistency: for every pair of constrained variables (Xi, Xj), every value
      in Xi's domain must have at least one supporting value in Xj's domain.
    - Run AC-3 before starting backtracking to reduce domains globally.
    - After each assignment, run AC-3 on arcs involving the assigned variable's neighbors.
    - If AC-3 empties any domain, the current assignment is inconsistent - backtrack.
    - You can create helper functions such as:
      - a values_compatible function to check if two variable-value pairs are consistent with the constraints.
      - a revise function that removes unsupported values from one variable's domain.
      - an ac3 function that manages the queue of arcs to check and calls revise.
      - a backtrack function that integrates AC-3 into the search process.
    """
    return None


def backtracking_mrv_lcv(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking with Forward Checking + MRV + LCV.

    Tips:
    - Combine the techniques from backtracking_fc, mrv_heuristic, and lcv_heuristic.
    - MRV (Minimum Remaining Values): Select the unassigned variable with the fewest legal values.
      Tie-break by degree: prefer the variable with the most unassigned neighbors.
    - LCV (Least Constraining Value): When ordering values for a variable, prefer
      values that rule out the fewest choices for neighboring variables.
    - Use csp.get_num_conflicts(var, value, assignment) to count how many values would be ruled out for neighbors if var=value is assigned.
    """
    # TODO: Implement your code here (BONUS)
    return None

