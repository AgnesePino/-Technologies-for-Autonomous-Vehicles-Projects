# Dijkstra benchmark on an OSM road network with travel-time edge costs.
# Counts expanded nodes on random pairs and visualizes the most-used edges.

import osmnx as ox
import random
import heapq

# Keep results reproducible
random.seed(31)

def generate_random_test_cases(graph, count=10):
    """Generate random pairs of distinct nodes for pathfinding tests."""
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

def dijkstra(orig, dest, plot=False):
    # Reset node state before each run
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        G.nodes[node]["distance"] = float("inf")
        G.nodes[node]["previous"] = None
        G.nodes[node]["previous_key"] = None
        G.nodes[node]["size"] = 0
    for edge in G.edges:
        style_unvisited_edge(edge)
    G.nodes[orig]["distance"] = 0
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50
    pq = [(0, orig)]
    step = 0
    while pq:
        _, node = heapq.heappop(pq)
        if node == dest:
            return step
        if G.nodes[node]["visited"]: continue
        G.nodes[node]["visited"] = True
        for u, v, k in G.out_edges(node, keys=True):
            style_visited_edge((u, v, k))
            neighbor = v
            weight = G.edges[(u, v, k)]["weight"]
            if G.nodes[neighbor]["distance"] > G.nodes[node]["distance"] + weight:
                G.nodes[neighbor]["distance"] = G.nodes[node]["distance"] + weight
                G.nodes[neighbor]["previous"] = node
                G.nodes[neighbor]["previous_key"] = k
                heapq.heappush(pq, (G.nodes[neighbor]["distance"], neighbor))
                for u2, v2, k2 in G.out_edges(neighbor, keys=True):
                    style_active_edge((u2, v2, k2))
        step += 1
    
    # No route found
    return None

def reconstruct_path(orig, dest, plot=False, algorithm=None, reset_colors=True):
    if reset_colors:
        for edge in G.edges:
            style_unvisited_edge(edge)
    dist_m = 0.0
    curr = dest
    while curr != orig:
        prev = G.nodes[curr]["previous"]
        prev_key = G.nodes[curr]["previous_key"]
        if prev is None:
            return None
        dist_m += G.edges[(prev, curr, prev_key)]["length"]
        style_path_edge((prev, curr, prev_key))
        if algorithm:
            G.edges[(prev, curr, prev_key)][f"{algorithm}_uses"] = G.edges[(prev, curr, prev_key)].get(f"{algorithm}_uses", 0) + 1
        curr = prev
    return dist_m

# Cities to benchmark
cities = ["Turin, Piedmont, Italy", "Aosta, Aosta, Italy"]

# Results stored per city
all_results = {}

for place_name in cities:
    # Load the road network for this city
    G = ox.graph_from_place(place_name, network_type="drive")
    for edge in G.edges:
        # Normalize maxspeed (OSM can store it as list/string/missing)
        maxspeed = 40
        if "maxspeed" in G.edges[edge]:
            maxspeed = G.edges[edge]["maxspeed"]
            if type(maxspeed) == list:
                speeds = [int(speed) if speed != "walk" else 1 for speed in maxspeed]
                maxspeed = min(speeds)
            elif type(maxspeed) == str:
                if maxspeed == "walk": 
                    maxspeed = 1
                else:
                    maxspeed = maxspeed.replace(" mph", "")
                    maxspeed = int(maxspeed)
        G.edges[edge]["maxspeed"] = maxspeed
        # Travel time in seconds
        G.edges[edge]["weight"] = G.edges[edge]["length"] * 3.6 / maxspeed
        G.edges[edge]["dijkstra_uses"] = 0
    
    num_nodes = len(G.nodes)
    num_edges = len(G.edges)
    
    # Run 10 random origin-destination pairs
    iterations = []
    travel_times = []
    distances_m = []
    completed_runs = 10
    nodes_list = list(G.nodes)
    test_number = 1
    while len(iterations) < completed_runs:
        start, end = random.sample(nodes_list, 2)
        it = dijkstra(start, end)
        if it is not None:
            iterations.append(it)
            travel_times.append(G.nodes[end]["distance"])
            dist_m = reconstruct_path(start, end, algorithm="dijkstra", plot=False, reset_colors=False)
            if dist_m is not None:
                distances_m.append(dist_m)
        test_number += 1
    
    if iterations:
        avg = sum(iterations) / len(iterations)
        avg_distance = sum(distances_m) / len(distances_m) if distances_m else 0.0
        avg_time = sum(travel_times) / len(travel_times) if travel_times else 0.0

        # Save city summary
        all_results[place_name] = {
            "nodes": num_nodes,
            "edges": num_edges,
            "all": iterations,
            "avg": avg,
            "avg_distance": avg_distance,
            "avg_time": avg_time
        }
    
# Final summary
for place_name, data in all_results.items():
    print(f"\n{'─'*70}")
    print(f"CITY: {place_name}")
    print(f"{'─'*70}")
    print(f"  Nodes: {data['nodes']}")
    print(f"  Edges: {data['edges']}\n")
    
    print("  Dijkstra:")
    for i, iters in enumerate(data['all'], 1):
        print(f"\tRun {i}: {iters} iterations")
    print(f"\tAverage Iterations:    {data['avg']:.2f}")
    print(f"\tAverage Distance:      {data['avg_distance']:.2f} m")
    print(f"\tAverage Travel Time:   {data['avg_time']:.2f} s")

# Plot only after all calculations and prints are done
for edge in G.edges:
    uses = G.edges[edge]["dijkstra_uses"]
    if uses > 0:
        G.edges[edge]["color"] = "blue"
        G.edges[edge]["alpha"] = min(0.4 + uses * 0.15, 1.0)
        G.edges[edge]["linewidth"] = 2 + uses * 1.5
    else:
        style_unvisited_edge(edge)

plot_graph()
