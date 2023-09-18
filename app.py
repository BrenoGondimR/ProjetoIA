import heapq
from flask import Flask, render_template, request, jsonify
from queue import PriorityQueue
from networkx import dfs_predecessors
import networkx as nx
import osmnx as ox
import psutil

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Função para verificar o uso atual da memória RAM
def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Memória em MB

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    start_point = data['start']
    end_point = data['end']
    algorithm = data['algorithm']
    memoriaStart = get_memory_usage()

    # Carregar um grafo do OpenStreetMap
    location = 'Fortaleza, Ceará, Brasil'
    try:
        G = ox.graph_from_place(location, network_type='drive')
    except Exception as e:
        print(f"Erro ao carregar o grafo: {e}")

    # Encontrar os nós mais próximos aos pontos de partida e chegada
    start_node = ox.distance.nearest_nodes(G, X=(start_point['lng']), Y=(start_point['lat']))
    end_node = ox.distance.nearest_nodes(G, X=(end_point['lng']), Y=(end_point['lat']))

    # Executar o algoritmo de busca selecionado
    path = []
    if algorithm == 'bfs':
        before_algo_memory = get_memory_usage()
        print(f"Memória antes de executar BFS: {before_algo_memory} MB")
        path = ox.distance.shortest_path(G, start_node, end_node, weight='length')
        after_algo_memory = get_memory_usage()
        print(f"Memória depois de executar BFS: {after_algo_memory} MB")
        print(f"Memória usada pelo BFS: {after_algo_memory - before_algo_memory} MB")

    elif algorithm == 'dfs':
        before_algo_memory = get_memory_usage()
        print(f"Memória antes de executar DFS: {before_algo_memory} MB")

        # Inicializa um dicionário de predecessores e uma pilha para DFS
        predecessors = {}
        stack = [(start_node, [start_node])]

        while stack:
            (vertex, path) = stack.pop()

            # Se o nó de destino é encontrado, saia do loop
            if vertex == end_node:
                path_found = True
                break

            for neighbor in G.neighbors(vertex):
                if neighbor not in set(path):
                    new_path = path + [neighbor]
                    stack.append((neighbor, new_path))
                    predecessors[neighbor] = vertex  # Atribua o nó atual como predecessor do vizinho

        # Reconstruir o caminho do nó de destino ao nó de origem
        if path_found:
            path = new_path  # 'new_path' conterá o caminho encontrado
        else:
            path = []

        after_algo_memory = get_memory_usage()
        print(f"Memória depois de executar DFS: {after_algo_memory} MB")
        print(f"Memória usada pelo DFS: {after_algo_memory - before_algo_memory} MB")

    elif algorithm == 'uniform_cost':
        before_algo_memory = get_memory_usage()
        print(f"Memória antes de executar Busca de Custo Uniforme: {before_algo_memory} MB")

        # Implementação do algoritmo de Busca de Custo Uniforme
        pq = [(0, start_node, [])]
        visited = set()
        while pq:
            (cost, current_node, path) = heapq.heappop(pq)
            if current_node not in visited:
                visited.add(current_node)
                if current_node == end_node:
                    path_found = True
                    break
                for neighbor, edge_data in G[current_node].items():
                    edge_weight = edge_data[0].get('length', 1)
                    heapq.heappush(pq, (cost + edge_weight, neighbor, path + [neighbor]))

        after_algo_memory = get_memory_usage()
        print(f"Memória depois de executar Busca de Custo Uniforme: {after_algo_memory} MB")

    elif algorithm == 'greedy':
        before_algo_memory = get_memory_usage()
        print(f"Memória antes de executar Busca Gulosa: {before_algo_memory} MB")

        # Implementação do algoritmo de Busca Gulosa
        end_coordinates = (G.nodes[end_node]['y'], G.nodes[end_node]['x'])
        pq = [(0, start_node, [])]
        visited = set()
        while pq:
            (heuristic, current_node, path) = heapq.heappop(pq)
            if current_node not in visited:
                visited.add(current_node)
                if current_node == end_node:
                    path_found = True
                    break
                for neighbor in G.neighbors(current_node):
                    neighbor_coordinates = (G.nodes[neighbor]['y'], G.nodes[neighbor]['x'])
                    heuristic = ox.distance.euclidean_dist_vec(end_coordinates[0], end_coordinates[1],
                                                               neighbor_coordinates[0], neighbor_coordinates[1])
                    heapq.heappush(pq, (heuristic, neighbor, path + [neighbor]))

        after_algo_memory = get_memory_usage()
        print(f"Memória depois de executar Busca Gulosa: {after_algo_memory} MB")

    elif algorithm == 'iddfs':
        before_algo_memory = get_memory_usage()
        print(f"Memória antes de executar Busca iterativa em profundidade: {before_algo_memory} MB")
        max_depth = 50  # profundidade máxima que você deseja procurar
        for depth in range(max_depth):
            stack = [(start_node, [start_node])]
            visited = set()
            while stack:
                (current_node, path) = stack.pop()
                if current_node in visited:
                    continue
                visited.add(current_node)
                if current_node == end_node:
                    break  # path contém o caminho encontrado
                if len(path) <= depth:
                    for neighbor in G.neighbors(current_node):
                        if neighbor not in visited:
                            new_path = path + [neighbor]
                            stack.append((neighbor, new_path))

        after_algo_memory = get_memory_usage()
        print(f"Memória depois de executar Busca Gulosa: {after_algo_memory} MB")

    elif algorithm == 'astar':
        before_algo_memory = get_memory_usage()
        print(f"Memória antes de executar A*: {before_algo_memory} MB")
        open_set = PriorityQueue()
        open_set.put((0, start_node))
        came_from = {}
        g_score = {node: float('inf') for node in G.nodes}
        g_score[start_node] = 0
        f_score = {node: float('inf') for node in G.nodes}
        f_score[start_node] = ox.distance.euclidean_dist_vec(G.nodes[start_node]['y'], G.nodes[start_node]['x'],
                                                             G.nodes[end_node]['y'], G.nodes[end_node]['x'])

        while not open_set.empty():
            current = open_set.get()[1]
            if current == end_node:
                path = []
                while current in came_from:
                    path.insert(0, current)
                    current = came_from[current]
                path.insert(0, start_node)
                break

            for neighbor in G.neighbors(current):
                tentative_g_score = g_score[current] + G[current][neighbor][0].get('length', 1)
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + ox.distance.euclidean_dist_vec(G.nodes[neighbor]['y'],
                                                                                           G.nodes[neighbor]['x'],
                                                                                           G.nodes[end_node]['y'],
                                                                                           G.nodes[end_node]['x'])
                    open_set.put((f_score[neighbor], neighbor))

        after_algo_memory = get_memory_usage()
        print(f"Memória depois de executar A*: {after_algo_memory} MB")

    total_memory_used = after_algo_memory - before_algo_memory
    path_info = []
    ordered_streets = []
    last_street = None

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        point_u = G.nodes[u]
        point_v = G.nodes[v]

        try:
            edge_data = G.get_edge_data(u, v)[0]
            street_name = edge_data.get('name', 'Unknown')
        except KeyError:
            street_name = 'Unknown'

        segment = {
            'coordinates': [[point_u['y'], point_u['x']], [point_v['y'], point_v['x']]],
            'street_name': street_name
        }
        path_info.append(segment)

        # Adicione o nome da rua à lista ordenada se for diferente do último
        if street_name != last_street:
            ordered_streets.append(street_name)
            last_street = street_name

    # Imprima as ruas em ordem
    print("Ruas percorridas em ordem:", ordered_streets)

    # Após calcular o caminho, use o seguinte código para plotar e salvar a rota:
    fig, ax = ox.plot.plot_graph_route(G, path, route_color='r', route_linewidth=2)
    fig.savefig('static/route.png')
    return jsonify({'path_info': path_info, 'image_url': 'static/route.png', 'ordered_streets': ordered_streets, 'total_memory_used_MB': total_memory_used})


if __name__ == '__main__':
    app.run(debug=True)
