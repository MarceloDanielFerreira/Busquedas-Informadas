##
# Proy#1: Resolucion de problemas con busqueda
# Estudiante: Marcelo Daniel Ferreira Fernández
import heapq
import time
from math import *


######
# UTILITARIOS
######

# Esto te sirve para calcular el tiempo
# Fuente: http://stackoverflow.com/questions/5998245/get-current-time-in-milliseconds-in-python
def current_time() :
    return int(round(time.time()*1000))


# Función para generar estados sucesores a partir de un estado inicial en un problema de búsqueda
def sucesores(camino_original):
    # Obtiene el estado actual del camino (último estado en la lista de caminos)
    estado_actual = camino_original[-1]
    
    # Lista donde se almacenarán los sucesores generados
    sucesores = []

    # Busca la posición del espacio vacío (representado por 0) en el estado actual
    # Recorre la matriz para encontrar el valor 0 y almacena su posición en fila y col
    fila, col = next((r, c) for r, fila in enumerate(estado_actual) for c, val in enumerate(fila) if val == 0)

    # Definición de los movimientos posibles (arriba, abajo, izquierda, derecha)
    movimientos = [(fila-1, col), (fila+1, col), (fila, col-1), (fila, col+1)]

    # Recorre cada movimiento posible para generar nuevos estados
    for r, c in movimientos:
        # Verifica si el movimiento está dentro de los límites de la matriz
        if 0 <= r < len(estado_actual) and 0 <= c < len(estado_actual[0]):
            # Crea una copia profunda del estado actual para modificarlo sin alterar el original
            nuevo_estado = [fila[:] for fila in estado_actual]
            
            # Realiza el intercambio del espacio vacío con la nueva posición
            nuevo_estado[fila][col], nuevo_estado[r][c] = nuevo_estado[r][c], nuevo_estado[fila][col]
            
            # Añade el nuevo estado al camino y lo agrega a la lista de sucesores
            sucesores.append(camino_original + [nuevo_estado])

    # Devuelve la lista de caminos que incluyen todos los estados sucesores generados
    return sucesores

######
# RESOLVEDOR GENERAL DE PROBLEMAS
######

# Función que implementa la búsqueda en anchura con costo uniforme (UC-BFS)
def uc_bfs(inicio, meta, sucesores_fn, costo_camino_fn=len, max_prof=1000):
    # La frontera se implementa como una heap de prioridad
    frontera = [(0, [inicio])]  # (costo acumulado, camino actual)
    visitados = set()  # Conjunto para almacenar los estados ya visitados
    num_estados_recorridos = 0  # Contador de estados explorados
    
    # Mientras haya estados en la frontera
    while frontera:
        # Extrae el estado con el costo acumulado más bajo
        costo, camino = heapq.heappop(frontera)
        estado_actual = camino[-1]  # Último estado en el camino actual
        
        # Convierte el estado actual a tupla para poder almacenarlo en el conjunto 'visitados'
        estado_actual_tuple = convertir_a_tupla(estado_actual)
        if estado_actual_tuple in visitados:
            # Si el estado ya fue visitado, lo ignora
            continue
        
        # Si el estado actual es el estado meta, retorna el camino y el número de estados recorridos
        if estado_actual == meta:
            return (num_estados_recorridos, camino)
        
        # Marca el estado actual como visitado
        visitados.add(estado_actual_tuple)
        num_estados_recorridos += 1  # Incrementa el contador de estados recorridos
        
        # Genera los estados sucesores del estado actual
        for nuevo_camino in sucesores_fn(camino, meta):
            # Calcula el nuevo costo del camino
            nuevo_costo = costo_camino_fn(nuevo_camino)
            # Si la profundidad del nuevo camino no excede el límite máximo
            if len(nuevo_camino) <= max_prof:
                # Añade el nuevo camino a la frontera con su costo asociado
                heapq.heappush(frontera, (nuevo_costo, nuevo_camino))
    
    # Si se agota la frontera sin encontrar la meta, retorna None
    return (num_estados_recorridos, None)

# Función auxiliar para convertir un estado (listado de listas) en una tupla inmutable
def convertir_a_tupla(estado):
    """Convierte listas a tuplas de forma recursiva."""
    if isinstance(estado, list):
        # Convierte recursivamente cada elemento de la lista en una tupla
        return tuple(convertir_a_tupla(elemento) for elemento in estado)
    return estado  # Si el estado no es una lista, se retorna tal cual


# Función que implementa el algoritmo A* (A-Estrella) para la búsqueda de la ruta óptima
def a_estrella(inicio, meta, heuristica_fn, sucesores_fn=sucesores, costo_camino_fn=len, max_prof=1000):
    # La frontera se implementa como una heap de prioridad, que contiene tuplas con (f, g, camino)
    # f = costo estimado total (g + h), g = costo real acumulado, camino = secuencia de estados
    frontera = [(heuristica_fn([inicio], meta), 0, [inicio])]
    
    # Conjunto para almacenar estados ya visitados, evitando ciclos y redundancias
    visitados = set()
    
    # Contador para llevar la cuenta del número de estados recorridos
    num_estados_recorridos = 0
    
    # Mientras haya estados en la frontera
    while frontera:
        # Extrae el estado con el menor valor de f (costo estimado total)
        _, costo, camino = heapq.heappop(frontera)
        estado_actual = camino[-1]  # Obtiene el último estado en el camino actual
        
        # Verifica si el estado actual es el estado meta
        if estado_actual == meta:
            # Si se alcanza el estado meta, retorna el número de estados recorridos y el camino
            return (num_estados_recorridos, camino)
        
        # Verifica si el estado actual ya fue visitado
        if tuple(map(tuple, estado_actual)) in visitados:
            continue  # Si ya fue visitado, se ignora y se pasa al siguiente estado
        
        # Marca el estado actual como visitado convirtiéndolo a tupla para añadirlo al conjunto
        visitados.add(tuple(map(tuple, estado_actual)))
        num_estados_recorridos += 1  # Incrementa el contador de estados recorridos
        
        # Genera los estados sucesores a partir del estado actual
        for nuevo_camino in sucesores_fn(camino, meta):
            # Calcula el nuevo costo acumulado g(n) del nuevo camino
            nuevo_costo = costo + costo_camino_fn(nuevo_camino)
            # Calcula el valor de f(n) = g(n) + h(n), donde h(n) es la heurística
            f = nuevo_costo + heuristica_fn(nuevo_camino, meta)
            
            # Si la profundidad del nuevo camino no excede el límite máximo
            if len(nuevo_camino) <= max_prof:
                # Añade el nuevo camino a la frontera con su costo total estimado f
                heapq.heappush(frontera, (f, nuevo_costo, nuevo_camino))
    
    # Si se agota la frontera sin encontrar la meta, retorna None
    return (num_estados_recorridos, None)




######
# FUNCIONES ESPECIFICAS A 8-PUZZLE 
######


def sucesores(camino_original,meta=None):
    # INGRESA TU CODIGO AQUI

    estado_actual = camino_original[-1]
    sucesores = []
    fila, col = next((r, c) for r, fila in enumerate(estado_actual) for c, val in enumerate(fila) if val == 0)

    movimientos = [(fila-1, col), (fila+1, col), (fila, col-1), (fila, col+1)]

    for r, c in movimientos:
        if 0 <= r < len(estado_actual) and 0 <= c < len(estado_actual[0]):
            nuevo_estado = [fila[:] for fila in estado_actual]
            nuevo_estado[fila][col], nuevo_estado[r][c] = nuevo_estado[r][c], nuevo_estado[fila][col]
            sucesores.append(camino_original + [nuevo_estado])


    # Devuelve una lista de caminos a tus soluciones
    return sucesores

# Función que calcula la distancia Manhattan total entre el estado actual y la meta
def manhattan(camino, meta):
    
    estado_actual = camino[-1]  # Obtiene el último estado en el camino actual
    distancia = 0  # Inicializa la distancia total en 0
    
    # Recorre cada celda del estado actual
    for r in range(len(estado_actual)):
        for c in range(len(estado_actual[r])):
            val = estado_actual[r][c]  # Obtiene el valor en la posición actual
            
            # Si el valor no es 0 (el espacio vacío no se considera)
            if val != 0:
                # Encuentra la posición (meta_r, meta_c) del valor 'val' en el estado meta
                meta_r, meta_c = next((i, j) for i, fila in enumerate(meta) for j, val_meta in enumerate(fila) if val_meta == val)
                
                # Calcula la distancia Manhattan entre la posición actual y la posición en la meta
                # La distancia Manhattan es la suma de las diferencias absolutas en las coordenadas de filas y columnas
                distancia += abs(r - meta_r) + abs(c - meta_c)
    
    # Devuelve la distancia Manhattan total para el último estado en el camino
    return distancia



# Función que calcula la distancia Euclidiana total entre el estado actual y la meta
def euclidiana(camino, meta):
    
    estado_actual = camino[-1]  # Obtiene el último estado en el camino actual
    distancia = 0  # Inicializa la distancia total en 0
    
    # Recorre cada celda del estado actual
    for r in range(len(estado_actual)):
        for c in range(len(estado_actual[r])):
            val = estado_actual[r][c]  # Obtiene el valor en la posición actual
            
            # Si el valor no es 0 (el espacio vacío no se considera)
            if val != 0:
                # Encuentra la posición (meta_r, meta_c) del valor 'val' en el estado meta
                meta_r, meta_c = next((i, j) for i, fila in enumerate(meta) for j, val_meta in enumerate(fila) if val_meta == val)
                
                # Calcula la distancia Euclidiana entre la posición actual y la posición en la meta
                # La distancia Euclidiana se calcula con la fórmula de la distancia entre dos puntos en un plano 2D
                distancia += sqrt((r - meta_r) ** 2 + (c - meta_c) ** 2)
    
    # Devuelve la distancia Euclidiana total para el último estado en el camino
    return distancia
 
# Función que calcula el número de "bad tiles" (o piezas mal ubicadas) entre el estado actual y la meta
def bad_tiles(camino, meta):
    estado_actual = camino[-1]  # Obtiene el último estado en el camino actual
    bad_tiles_count = 0  # Inicializa el contador de piezas mal ubicadas en 0
    
    # Recorre cada celda del estado actual
    for r in range(len(estado_actual)):
        for c in range(len(estado_actual[r])):
            # Si la pieza no es el espacio vacío (0) y no está en la posición correcta según la meta
            if estado_actual[r][c] != 0 and estado_actual[r][c] != meta[r][c]:
                bad_tiles_count += 1  # Incrementa el contador de piezas mal ubicadas
    
    # Devuelve el número total de piezas mal ubicadas en el último estado del camino
    return bad_tiles_count

#Propia
def manhattan_inversions(camino, meta):
    # Función auxiliar para contar las inversiones en una lista plana de valores
    def count_inversions(state):
        # Aplana el estado en una lista de valores, excluyendo el valor 0 (espacio vacío)
        flat_state = [val for row in state for val in row if val != 0]
        inversions = 0  # Inicializa el contador de inversiones en 0
        
        # Recorre cada par de valores en la lista aplanada
        for i in range(len(flat_state)):
            for j in range(i + 1, len(flat_state)):
                # Incrementa el contador de inversiones si el valor en la posición i es mayor que en la posición j
                if flat_state[i] > flat_state[j]:
                    inversions += 1
        return inversions
    
    # Función auxiliar para calcular la distancia Manhattan entre el estado actual y la meta
    def manhattan_distance(state, meta):
        distancia = 0  # Inicializa la distancia total en 0
        for r in range(len(state)):
            for c in range(len(state[r])):
                val = state[r][c]  # Obtiene el valor en la posición actual
                if val != 0:  # Ignora el espacio vacío (0)
                    # Encuentra la posición (meta_r, meta_c) del valor 'val' en el estado meta
                    meta_r, meta_c = next((i, j) for i, fila in enumerate(meta) for j, val_meta in enumerate(fila) if val_meta == val)
                    # Calcula la distancia Manhattan entre la posición actual y la posición en la meta
                    distancia += abs(r - meta_r) + abs(c - meta_c)
        return distancia
    
    # Obtener el último estado del camino
    estado_actual = camino[-1]
    
    # Calcular la distancia Manhattan entre el estado actual y la meta
    distancia_manhattan = manhattan_distance(estado_actual, meta)
    
    # Contar las inversiones en el estado actual
    inversiones = count_inversions(estado_actual)
    
    # Ajustar la heurística con una constante de penalización para el número de inversiones
    PENALIZACION_INVERSIONES = 2  
    
    # Combinar las dos heurísticas (distancia Manhattan y penalización por inversiones)
    heuristica = distancia_manhattan + PENALIZACION_INVERSIONES * inversiones
    
    # Devuelve la heurística combinada
    return heuristica



   
def solvable(estado) :
    invs = 0
    for a in range(len(estado)**2) :
        for b in range(a+1, len(estado)**2) :
             if estado[a/len(estado)][a%len(estado)] == estado[b/len(estado)][b%len(estado)] :
                 invs +=1
    return invs%2 == 0


######
# FUNCIONES ESPECIFICAS A ARAD-BUCAREST
######
def costo_camino(camino):
    costo_total = 0  # Inicializa el costo total en 0
    
    # Itera sobre las ciudades en el camino, sumando las distancias entre ciudades consecutivas
    for i in range(len(camino) - 1):
        ciudad_actual = camino[i]  # Ciudad en la posición actual del camino
        ciudad_siguiente = camino[i + 1]  # Ciudad en la posición siguiente del camino
        
        # Obtener la distancia entre la ciudad actual y la siguiente
        for (ciudad, distancia) in adyacentes_romania(ciudad_actual):
            if ciudad == ciudad_siguiente:  # Verifica si la ciudad es la siguiente en el camino
                costo_total += distancia  # Suma la distancia al costo total
                break  # Sale del bucle una vez encontrada la distancia
    
    return costo_total  # Devuelve el costo total del camino

def suc_ciudades(camino, meta=None):
    # Obtener la ciudad actual (la última ciudad en el camino)
    ciudad_actual = camino[-1]
    
    # Obtener las ciudades adyacentes y sus distancias desde la ciudad actual
    adyacentes = adyacentes_romania(ciudad_actual)
    
    # Generar nuevos caminos al agregar cada ciudad adyacente
    sucesores = []
    for (ciudad, distancia) in adyacentes:
        nuevo_camino = camino + [ciudad]  # Crea un nuevo camino añadiendo la ciudad adyacente
        sucesores.append(nuevo_camino)  # Añade el nuevo camino a la lista de sucesores
    
    return sucesores  # Devuelve la lista de nuevos caminos posibles

def dist_ciudades(camino, meta):
    ciudad_actual = camino[-1]  # Obtiene la ciudad actual del camino
    adyacentes = adyacentes_romania(ciudad_actual)  # Obtiene las ciudades adyacentes
    
    # Busca la distancia directa desde la ciudad actual a la meta
    for (ciudad, distancia) in adyacentes:
        if ciudad == meta:  # Verifica si la ciudad adyacente es la meta
            return distancia  # Devuelve la distancia encontrada
    
    # Si la ciudad no es adyacente a la meta, retorna una distancia alta (infinito)
    return float('inf')

def heuristica_efectiva(camino, meta):
    ciudad_actual = camino[-1]  # Obtiene la ciudad actual del camino
    
    # Distancia heurística directa desde la ciudad actual a Bucarest
    dist_directa = dist_a_bucarest(ciudad_actual)
    
    # Buscar la distancia mínima a Bucarest a través de las ciudades adyacentes
    adyacentes = adyacentes_romania(ciudad_actual)
    dist_min_adyacente = float('inf')  # Inicializa la distancia mínima a infinito
    for (ciudad, distancia) in adyacentes:
        dist_a_bucarest_adyacente = dist_a_bucarest(ciudad)
        if dist_a_bucarest_adyacente < dist_min_adyacente:
            dist_min_adyacente = dist_a_bucarest_adyacente  # Actualiza la distancia mínima si es menor

    # Combina la distancia directa con la mínima distancia a través de las ciudades adyacentes
    heuristica = min(dist_directa, dist_min_adyacente)
    
    return heuristica  # Devuelve la heurística efectiva

# Distancia Euclidiana entre la ciudad ingresada y Bucarest
def dist_a_bucarest(ciudad, meta=None):
    ROMANIA_EUC = {
        'arad': 366, 'bucarest': 0, 'craiova': 160, 'dobreta': 242, 'eforie': 161, 
        'fagaras': 176, 'giurgiu': 77, 'hirsova': 151, 'iasi': 226, 'lugoj': 244, 'mehadia': 241,
        'neamt': 234, 'oradea': 380, 'pitesti': 100, 'rimnicu vilcea': 193, 'sibiu': 253, 'timisoara': 329,
        'urziceni': 80, 'vaslui': 199, 'zerind': 374
    }
    return ROMANIA_EUC.get(ciudad, float('inf'))

# Esto es basicamente una lista de adyacencia (un grafo) con todas las distancias entre ciudades
# (suponiendo que vas en tren o auto las distancias directas entre ciudades podrian ser mas cortas)
def adyacentes_romania(ciudad) :
    ROMANIA_ADY = { 'arad': (('zerind',75),('timisoara',118),('sibiu', 99)),
                    'timisoara': (('arad',118),('lugoj',111)),
                    'lugoj':(('timisoara',111),('mehadia',70)),
                    'mehadia':(('lugoj',70),('dobreta',75)),
                    'dobreta':(('mehadia',75),('craiova',120)),
                    'craiova':(('dobreta',120),('rimnicu vilcea', 146),('pitesti', 138)),
                    'pitesti':(('craiova',138),('rimnicu vilcea', 97),('bucarest',101)),
                    'rimnicu vilcea':(('craiova',146),('pitesti',97),('sibiu',80)),
                    'sibiu':(('arad',140),('oradea',151),('fagaras',99),('rimnicu vilcea', 80)),
                    'zerind':(('arad',75),('oradea',71)),
                    'fagaras':(('sibiu',99),('bucarest',211)),
                    'bucarest':(('giurgui',90),('pitesti',101),('fagaras',211),('urziceni',85)),
                    'urziceni':(('bucarest',85),('hirsova',98),('vaslui',142)),
                    'hirsova':(('urziceni',98),('eforie',86)),
                    'vaslui':(('urziceni',142),('iasi',92)),
                    'iasi':(('vaslui',92),('neamt',87)),
                    'oradea':(('zerind',71),('sibiu',151)) }
    return ROMANIA_ADY[ciudad] 



#######################################
# Tests
#######################################

meta = [[1,2,3],[4,5,6],[7,8,0]]
inicio = [[0,1,2],[3,4,5],[6,7,8]]


print(manhattan([inicio], meta))
print(manhattan_inversions([inicio], meta))
print(euclidiana([inicio], meta))
print(bad_tiles([inicio], meta))





print('UC-BFS 8-PUZZlLE')
begin = current_time()
solucion = uc_bfs(inicio, meta, sucesores)
print(solucion, current_time()-begin)
print('A* Propia 8-Puzzle')
begin = current_time()
solucion = a_estrella(inicio,meta, manhattan_inversions, sucesores)
print(solucion, current_time()-begin)

print('A* Manhattan 8-Puzzle')
begin = current_time()
solucion = a_estrella(inicio,meta, manhattan, sucesores)
print(solucion, current_time()-begin)

print('A* Bad Tiles 8-Puzzle')
begin = current_time()
solucion = a_estrella(inicio,meta, bad_tiles, sucesores)
print(solucion, current_time()-begin)
print('A* Euclidoana 8-Puzzle')
begin = current_time()
solucion = a_estrella(inicio,meta, euclidiana, sucesores)
print(solucion, current_time()-begin)

print('UC-BFS ciudades')
begin = current_time()
solucion = uc_bfs('arad', 'bucarest', suc_ciudades, costo_camino)
print(solucion, current_time()-begin)

print('A* Dist Eculidiana Ciudades')
begin = current_time()
solucion = a_estrella('arad', 'bucarest', heuristica_efectiva, suc_ciudades, costo_camino)
print(solucion, current_time()-begin)

