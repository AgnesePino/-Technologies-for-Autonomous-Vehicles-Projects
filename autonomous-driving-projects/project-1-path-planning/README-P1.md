# Global Path Planning: Dijkstra vs A*

This project compares **Dijkstra** and **A\*** algorithms for global path planning on real road networks extracted from **OpenStreetMap**. The goal is to evaluate how efficiently each algorithm finds the shortest route between two points in an urban road graph.

Global path planning can be modeled as a shortest-path problem on a graph, where:

- nodes represent road intersections;
- edges represent road segments;
- edge weights represent the estimated travel time needed to traverse each road segment.

The project focuses on the comparison between an uninformed search strategy, Dijkstra, and an informed search strategy, A*, using different heuristic functions.

## Project Overview

Two road networks with different sizes are considered:

- **Turin**, a large and dense urban network;
- **Aosta**, a smaller road network.

For each city, the same randomly generated start-goal pairs are used for all algorithms in order to make the comparison fair.

The algorithms tested are:

- **Dijkstra**;
- **A\*** with Manhattan heuristic;
- **A\*** with Euclidean heuristic;
- **A\*** with Haversine heuristic.

The main performance metric is the number of expanded nodes, also called iterations. A lower number of iterations means that the algorithm reaches the destination while exploring a smaller part of the graph.

## Algorithms

### Dijkstra

Dijkstra's algorithm finds the shortest path by always expanding the node with the smallest accumulated cost from the start node. During the search, the distances to neighboring nodes are updated whenever a shorter path is found.

Since Dijkstra does not use any information about the goal position, it can explore many nodes before reaching the destination, especially in large graphs.

### A*

A* improves the search by adding a heuristic estimate of the remaining cost to the goal. Nodes are selected according to:

```text
f(n) = g(n) + h(n)
```

where:

- `g(n)` is the cost from the start node to the current node;
- `h(n)` is the estimated cost from the current node to the goal.

If the heuristic is admissible, A* can still find the optimal path while usually expanding fewer nodes than Dijkstra.

## Heuristics

Three heuristic functions are implemented and compared:

### Manhattan distance

The Manhattan distance is based on the sum of the horizontal and vertical differences between two points.

This heuristic can guide the search very directly and often reduces the number of explored nodes. However, it is not always admissible on real road networks because roads are not arranged only along horizontal and vertical directions. For this reason, it can sometimes produce slightly longer paths.

### Euclidean distance

The Euclidean distance is the straight-line distance between two points. It is generally a good estimate of the remaining distance in a local road network and preserves optimality in the tested cases.

### Haversine distance

The Haversine distance computes the distance between two geographic points on the Earth's surface. At the scale of a city, its behavior is very similar to the Euclidean distance.

## Implementation Details

The project is implemented in **Python** using **OSMnx** to download and build road graphs from OpenStreetMap.

Each road network is represented as a directed graph:

- nodes are intersections or relevant road points;
- edges are road segments;
- edge costs are based on estimated travel time.

The cost of each edge is computed as:

```text
cost = length * 3.6 / speed
```

where:

- `length` is the edge length in meters;
- `speed` is the road speed in km/h;
- the resulting cost is expressed in seconds.

Both Dijkstra and A* use a priority queue implemented with Python's `heapq` module. Outdated entries in the queue are ignored during the search.

For A*, geometric distances are converted into estimated travel times by dividing the heuristic distance by a reference speed. The reference speed is chosen as the maximum valid `maxspeed` value available in the road network. If no valid value is found, a fallback speed of 130 km/h is used.

## Experimental Setup

For each city, ten random start-goal node pairs are generated. The same pairs are used for all algorithms and heuristics.

For each run, the following metrics are measured:

- number of expanded nodes;
- total path distance;
- total travel time.

The final results are reported as average values over the ten runs.

## Results

### Turin

The Turin road network contains:

- **11,768 nodes**;
- **25,206 edges**.

Because the graph is relatively large, the advantage of A* becomes clear. All A* variants expand fewer nodes than Dijkstra while producing the same path distance and travel time.

| Algorithm | Avg Iterations | Avg Distance (m) | Avg Time (s) |
| --- | ---: | ---: | ---: |
| Dijkstra | 6318.00 | 6031.45 | 469.91 |
| A* Manhattan | 3430.30 | 6031.45 | 469.91 |
| A* Euclidean | 3982.20 | 6031.45 | 469.91 |
| A* Haversine | 3979.20 | 6031.45 | 469.91 |

In Turin, A* reduces the number of node expansions by about **37-46%** compared to Dijkstra. The Manhattan heuristic gives the lowest number of iterations, while Euclidean and Haversine produce almost identical results.

### Aosta

The Aosta road network contains:

- **825 nodes**;
- **1,723 edges**.

Since this network is much smaller, all algorithms require fewer node expansions. However, A* still explores fewer nodes than Dijkstra.

| Algorithm | Avg Iterations | Avg Distance (m) | Avg Time (s) |
| --- | ---: | ---: | ---: |
| Dijkstra | 343.00 | 2380.47 | 203.23 |
| A* Manhattan | 163.60 | 2383.55 | 203.49 |
| A* Euclidean | 201.70 | 2380.47 | 203.23 |
| A* Haversine | 201.50 | 2380.47 | 203.23 |

In Aosta, A* reduces the number of node expansions by about **41-52%**. Euclidean and Haversine preserve the same path distance and travel time as Dijkstra. Manhattan expands fewer nodes, but produces a slightly longer path in some cases.

### Average Iterations Comparison

| Algorithm | Turin | Aosta |
| --- | ---: | ---: |
| Dijkstra | 6318.00 | 343.00 |
| A* Manhattan | 3430.30 | 163.60 |
| A* Euclidean | 3982.20 | 201.70 |
| A* Haversine | 3979.20 | 201.50 |

## Main Findings

The results show that A* is generally more efficient than Dijkstra for path planning on road networks. The use of a heuristic allows A* to explore a smaller part of the graph before reaching the destination.

The main observations are:

- A* expands fewer nodes than Dijkstra in all tested cases;
- Euclidean and Haversine heuristics preserve the optimal solution in the experiments;
- Manhattan is more aggressive and often expands the fewest nodes;
- Manhattan can sometimes produce a slightly longer path because it is not always admissible on real road networks;
- the difference between Euclidean and Haversine is negligible at city scale;
- the advantage of A* is more evident on larger graphs.

## Technologies Used

- Python
- OSMnx
- NetworkX
- OpenStreetMap data
- heapq
- Matplotlib

## Conclusion

This project demonstrates the use of classical shortest-path algorithms for autonomous driving path planning. Dijkstra guarantees the shortest path but can require many node expansions because it does not use goal-directed information. A* uses heuristics to guide the search and significantly reduces the computational effort.

The choice of heuristic is important. Euclidean and Haversine provide a good balance between efficiency and optimality, while Manhattan can be faster but may slightly compromise optimality in some road networks.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
