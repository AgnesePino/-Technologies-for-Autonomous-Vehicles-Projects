# A* benchmark with three heuristics (Manhattan, Euclidean, Haversine)
# on a real OSM road network using travel time as edge cost.

import osmnx as ox
import random
import heapq
import math

# Keep results reproducible
random.seed(31)

def generate_random_test_cases(graph, count=10):
    """Generate random pairs of distinct nodes for testing pathfinding algorithms."""
    test_cases = []
    nodes_list = list(graph.nodes)
    
    for _ in range(count):
        start, end = random.sample(nodes_list, 2)
        test_cases.append((start, end))
    
    return test_cases

def style_unvisited_edge(edge):        
    G.edges[edge]["color"] = "gray"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 0.2

def style_visited_edge(edge):
    G.edges[edge]["color"] = "green"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_active_edge(edge):
    G.edges[edge]["color"] = "red"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_path_edge(edge):
    G.edges[edge]["color"] = "white"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 5

def plot_graph():
    ox.plot_graph(
        G,
        node_size =  [ G.nodes[node]["size"] for node in G.nodes ],
        edge_color = [ G.edges[edge]["color"] for edge in G.edges ],
        edge_alpha = [ G.edges[edge]["alpha"] for edge in G.edges ],
        edge_linewidth = [ G.edges[edge]["linewidth"] for edge in G.edges ],
        node_color = "white",
        bgcolor = "black"
    )

# Speed used in heuristic time estimates; updated per city from graph data.
MAX_SPEED = 130

def manhattan_distance(node1, node2):
    """Manhattan distance heuristic with latitude correction."""
    x1 = G.nodes[node1]["x"]
    y1 = G.nodes[node1]["y"]
    x2 = G.nodes[node2]["x"]
    y2 = G.nodes[node2]["y"]
    
    # Approximate lat/lon to meters
    avg_lat = (y1 + y2) / 2
    lat_dist_m = abs(y1 - y2) * 111000
    lon_dist_m = abs(x1 - x2) * 111000 * math.cos(math.radians(avg_lat))
    distance_m = lat_dist_m + lon_dist_m
    
    estimated_time = distance_m / (MAX_SPEED / 3.6)
    return estimated_time

def euclidean_distance(node1, node2):
    """Euclidean distance heuristic with latitude correction."""
    x1 = G.nodes[node1]["x"]
    y1 = G.nodes[node1]["y"]
    x2 = G.nodes[node2]["x"]
    y2 = G.nodes[node2]["y"]
    
    # Straight-line estimate on a local planar approximation
    avg_lat = (y1 + y2) / 2
    lat_dist_m = (y1 - y2) * 111000
    lon_dist_m = (x1 - x2) * 111000 * math.cos(math.radians(avg_lat))
    distance_m = math.sqrt(lat_dist_m**2 + lon_dist_m**2)
    
    estimated_time = distance_m / (MAX_SPEED / 3.6)
    return estimated_time

def haversine_distance(node1, node2):
    lat1 = G.nodes[node1]["y"]
    lon1 = G.nodes[node1]["x"]
    lat2 = G.nodes[node2]["y"]
    lon2 = G.nodes[node2]["x"]
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    R_m = 6371000
    distance_m = R_m * c
    speed_ms = MAX_SPEED / 3.6
    return distance_m / speed_ms

def astar(orig, dest, heuristic_func):
    """A* pathfinding algorithm using the provided heuristic function."""
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        G.nodes[node]["distance"] = float("inf")
        G.nodes[node]["previous"] = None
        G.nodes[node]["previous_key"] = None
        G.nodes[node]["size"] = 0

    for edge in G.edges:
        style_unvisited_edge(edge)

    G.nodes[orig]["distance"] = 0.0
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50

    h0 = heuristic_func(orig, dest)
    pq = [(h0, 0.0, orig)]
    step = 0

    while pq:
        f, g, node = heapq.heappop(pq)

        # Ignore outdated queue entries
        if g > G.nodes[node]["distance"]:
            continue

        if node == dest:
            return step

        if G.nodes[node]["visited"]:
            continue

        G.nodes[node]["visited"] = True
        step += 1

        for u, v, k in G.out_edges(node, keys=True):
            style_visited_edge((u, v, k))

            neighbor = v
            weight = G.edges[(u, v, k)]["weight"]

            new_distance = G.nodes[node]["distance"] + weight

            if new_distance < G.nodes[neighbor]["distance"]:
                G.nodes[neighbor]["distance"] = new_distance
                G.nodes[neighbor]["previous"] = node
                G.nodes[neighbor]["previous_key"] = k

                g_new = new_distance
                h_new = heuristic_func(neighbor, dest)
                f_new = g_new + h_new

                heapq.heappush(pq, (f_new, g_new, neighbor))

                for u2, v2, k2 in G.out_edges(neighbor, keys=True):
                    style_active_edge((u2, v2, k2))

    return None


def reconstruct_path(orig, dest, algorithm=None, reset_colors=True):
    if reset_colors:
        for edge in G.edges:
            style_unvisited_edge(edge)

    dist_m = 0.0
    speeds_ms = []
    curr = dest

    while curr != orig:
        prev = G.nodes[curr]["previous"]
        prev_key = G.nodes[curr]["previous_key"]

        if prev is None:
            return None, None

        edge_data = G.edges[(prev, curr, prev_key)]

        dist_m += edge_data["length"]
        speeds_ms.append(edge_data["maxspeed_ms"])

        style_path_edge((prev, curr, prev_key))

        if algorithm:
            key = f"{algorithm}_uses"
            edge_data[key] = edge_data.get(key, 0) + 1

        curr = prev

    return dist_m, speeds_ms


def parse_maxspeed(value, default=40):
    """Normalize OSM maxspeed into a numeric km/h value."""
    if value is None:
        return float(default) if default is not None else None

    if isinstance(value, list):
        parsed = [parse_maxspeed(v, default) for v in value]
        parsed = [v for v in parsed if v is not None]
        return min(parsed) if parsed else (float(default) if default is not None else None)

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        value = value.strip().lower()

        if value == "walk":
            return 1.0

        if ";" in value:
            parts = value.split(";")
            parsed = [parse_maxspeed(part.strip(), default) for part in parts]
            parsed = [v for v in parsed if v is not None]
            return min(parsed) if parsed else (float(default) if default is not None else None)

        if "mph" in value:
            try:
                mph_value = float(value.replace("mph", "").strip())
                return mph_value * 1.60934
            except ValueError:
                return float(default) if default is not None else None

        try:
            return float(value)
        except ValueError:
            return float(default) if default is not None else None

    return float(default) if default is not None else None


def get_graph_maxspeed_kmh(graph, fallback=130.0):
    """Return the maximum normalized maxspeed found in the graph (km/h)."""
    speeds = []

    for edge in graph.edges:
        value = graph.edges[edge].get("maxspeed", None)
        parsed = parse_maxspeed(value, default=None)
        if parsed is not None and parsed > 0:
            speeds.append(parsed)

    return max(speeds) if speeds else float(fallback)


# Cities to benchmark
cities = ["Turin, Piedmont, Italy", "Aosta, Aosta, Italy"]

# Results stored per city
all_results = {}

for place_name in cities:
    # Load the road network for the current city
    G = ox.graph_from_place(place_name, network_type="drive")

    for edge in G.edges:
        raw_maxspeed = G.edges[edge].get("maxspeed", 40)
        maxspeed_kmh = parse_maxspeed(raw_maxspeed, default=40)
        maxspeed_ms = maxspeed_kmh / 3.6

        G.edges[edge]["maxspeed"] = maxspeed_kmh
        G.edges[edge]["maxspeed_ms"] = maxspeed_ms

        # Edge cost = travel time (seconds)
        G.edges[edge]["weight"] = G.edges[edge]["length"] / maxspeed_ms

    MAX_SPEED = get_graph_maxspeed_kmh(G, fallback=130.0)

    num_nodes = len(G.nodes)
    num_edges = len(G.edges)

    heuristics = {
        "Manhattan": manhattan_distance,
        "Euclidean": euclidean_distance,
        "Haversine": haversine_distance
    }

    nodes_list = list(G.nodes)

    # Same 10 pairs for each heuristic
    test_cases = []
    for _ in range(10):
        start, end = random.sample(nodes_list, 2)
        test_cases.append((start, end))

    city_results = {}

    # Run A* with each heuristic on the same test cases
    for heuristic_name, heuristic_func in heuristics.items():
        for edge in G.edges:
            G.edges[edge][f"astar_{heuristic_name}_uses"] = 0

        iterations = []
        travel_times = []
        distances_m = []

        for start, end in test_cases:
            it = astar(start, end, heuristic_func)
            if it is not None:
                iterations.append(it)
                travel_times.append(G.nodes[end]["distance"])

                dist_m, _ = reconstruct_path(
                    start,
                    end,
                    algorithm=f"astar_{heuristic_name}",
                    reset_colors=False
                )
                if dist_m is not None:
                    distances_m.append(dist_m)

        if iterations:
            avg = sum(iterations) / len(iterations)
            avg_distance = sum(distances_m) / len(distances_m) if distances_m else 0.0
            avg_time = sum(travel_times) / len(travel_times) if travel_times else 0.0
            city_results[heuristic_name] = {
                "avg": avg,
                "all": iterations,
                "avg_distance": avg_distance,
                "avg_time": avg_time
            }

    all_results[place_name] = {
        "nodes": num_nodes,
        "edges": num_edges,
        "heuristics": city_results
    }

# Final summary
for place_name, data in all_results.items():
    print(f"\n{'─' * 70}")
    print(f"CITY: {place_name}")
    print(f"{'─' * 70}")
    print(f"  Nodes: {data['nodes']}")
    print(f"  Edges: {data['edges']}\n")

    print("  Heuristic Comparison:")
    for heuristic_name in ["Manhattan", "Euclidean", "Haversine"]:
        if heuristic_name in data["heuristics"]:
            heur_data = data["heuristics"][heuristic_name]
            iterations_list = heur_data["all"]
            avg = heur_data["avg"]
            avg_distance = heur_data["avg_distance"]
            avg_time = heur_data["avg_time"]

            print(f"    {heuristic_name}:")
            for i, iterations in enumerate(iterations_list, 1):
                print(f"\tRun {i}: {iterations} iterations")
            print(f"\tAverage Iterations:    {avg:.2f}")
            print(f"\tAverage Distance:      {avg_distance:.2f} m")
            print(f"\tAverage Travel Time:   {avg_time:.2f} s")
    print()

# Plot edge usage for the last processed city (Haversine)
for edge in G.edges:
    uses = G.edges[edge].get("astar_Haversine_uses", 0)
    if uses > 0:
        G.edges[edge]["color"] = "blue"
        G.edges[edge]["alpha"] = min(0.4 + uses * 0.15, 1.0)
        G.edges[edge]["linewidth"] = 2 + uses * 1.5
    else:
        style_unvisited_edge(edge)

plot_graph()