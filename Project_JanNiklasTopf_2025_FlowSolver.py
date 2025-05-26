"""
Flow Network Solver
--------------------

This module defines a `FlowNetwork` class to model a directed flow network with costs and capacities,
and provides implementations of:

1. Maximum Flow using a capacity-aware Dijkstra (widest path) algorithm.
2. Minimum-Cost Maximum-Flow (Successive Shortest Path with Reduced Costs and Potentials).
3. Min-Cut detection after computing max flow.
4. DOT/PDF graph visualizations of residual networks.

Classes:
- Edge: Represents a directed edge with capacity, cost, and flow.
- FlowNetwork: Builds and manipulates the flow network from input file, and runs the algorithms.

DISCLAIMER:
Some of the comments and documentation in this code were generated with the assistance of ChatGPT.
The purpose of using AI-generated explanations is to enhance clarity, maintainability, and understanding
of the algorithms implemented. While the core logic have been developed manually, the comments are reviewed
to ensure correctness and relevance.

"""

import heapq
from collections import deque, defaultdict
import subprocess


class Edge:
    """
    Represents a directed edge in a flow network.

    Attributes:
        u (int): The starting node of the edge.
        v (int): The ending node of the edge.
        capacity (int): The maximum capacity of the edge.
        cost (int): The cost per unit of flow for this edge.
        flow (int): The current amount of flow through the edge.
    """

    def __init__(self, u, v, capacity, cost):
        """
        Initializes an edge with a given start node, end node, capacity, and cost.

        Args:
            u (int): The start node of the edge.
            v (int): The end node of the edge.
            capacity (int): The maximum flow capacity of the edge.
            cost (int): The cost per unit of flow.
        """
        self.u = u                  # Start node of the edge
        self.v = v                  # End node of the edge
        self.capacity = capacity    # Maximum capacity of the edge
        self.cost = cost            # Cost per unit of flow
        self.flow = 0               # Initial flow is 0

    def residual_capacity(self):
        """
        Returns the residual (remaining) capacity of the edge.

        Returns:
            int: The amount of additional flow that can be pushed through this edge.
        """
        return self.capacity - self.flow


class FlowNetwork:
    """
    Implements a flow network supporting algorithms like max-flow and min-cost max-flow.

    Attributes:
        show_iterations (bool): If True, enables visualization after each algorithm iteration.
        graph (defaultdict): Adjacency list mapping each node to a list of outgoing edges.
        edges (list): List of all forward (original) edges in the network.
        input_file (str): Filename from which the graph was initially loaded.
    """

    def __init__(self, filename, show_itreations=False):
        """
        Initializes the flow network from a given input file.

        Args:
            filename (str): Path to the input file containing the graph structure.
            show_itreations (bool): Whether to enable visualization after each iteration (default: False).
        """
        self.show_iterations = show_itreations   # Whether to generate visualizations after each algorithm iteration
        self.graph = defaultdict(list)           # Adjacency list: maps nodes to a list of outgoing edges
        self.edges = []                          # List of all forward (original) edges
        self.input_file = filename               # Store the filename for possible reload/reset
        self.read_graph(filename)                # Read and initialize the graph from the file

    def read_graph(self, filename):
        """
        Reads the input graph from a file and constructs the internal representation.
        Handles conflicting (bidirectional) edges by introducing intermediate nodes to 
        avoid ambiguity in residual graph construction.

        The file format is expected as follows:
            Line 1: num_nodes num_arcs source sink
            Next num_arcs lines: u v capacity cost

        Args:
            filename (str): Path to the input file describing the graph.
        """
        with open(filename, 'r') as f:
            # Read the header line: number of nodes, number of arcs, source node, sink node
            self.num_nodes, num_arcs, self.source, self.sink = map(int, f.readline().split())
            self.original_nodes = self.num_nodes  # Store the original number of nodes (before adding intermediates)
            self.extra_node_id = self.num_nodes   # ID to assign to new intermediate nodes
            edge_pairs = set()  # Used to track edge directions and detect bidirectional (conflicting) edges

            for _ in range(num_arcs):
                u, v, cap, cost = map(int, f.readline().split())

                # If the reverse edge (v -> u) was already added, we have a conflict
                # Introduce an intermediate node to separate the two directions
                if (v, u) in edge_pairs:
                    intermediate = self.extra_node_id
                    self.extra_node_id += 1
                    self.num_nodes += 1

                    # Split the edge into two: u -> intermediate and intermediate -> v
                    self.add_edge(u, intermediate, cap, cost)
                    self.add_edge(intermediate, v, cap, 0)  # No cost on the second half
                else:
                    self.add_edge(u, v, cap, cost)
                    edge_pairs.add((u, v))

        # Add an artificial edge from sink to source with zero capacity and cost
        # Useful for algorithms that work with cycles (e.g. Min-Cost Max-Flow)
        self.add_edge(self.sink, self.source, 0, 0)

    def add_edge(self, u, v, capacity, cost):
        """
        Adds a forward edge and its corresponding residual (backward) edge to the network.

        Args:
            u (int): Start node of the edge.
            v (int): End node of the edge.
            capacity (int): Capacity of the forward edge.
            cost (int): Cost per unit of flow on the forward edge.
        """
        forward = Edge(u, v, capacity, cost)   # Actual edge from u to v
        backward = Edge(v, u, 0, -cost)        # Residual edge with 0 capacity and negative cost

        # Link the two edges so we can easily access the reverse edge when augmenting
        forward.rev = backward
        backward.rev = forward

        # Add edges to the adjacency list
        self.graph[u].append(forward)
        self.graph[v].append(backward)

        # Keep track of forward edges only (for reporting / statistics)
        self.edges.append(forward)

    def get_min_cut(self):
        """
        Computes the minimum cut of the flow network after running a max-flow algorithm.

        The method performs a BFS from the source over edges with positive residual capacity.
        The set of reachable nodes forms one side of the minimum cut. The cut-set consists
        of all edges going from reachable nodes to non-reachable nodes.

        Returns:
            visited (set): Set of nodes reachable from the source in the residual graph.
            min_cut_edges (list): List of tuples (u, v, capacity, cost) representing the edges
                                  that cross the min-cut (i.e., from visited to non-visited nodes).
        """
        visited = set()
        queue = deque([self.source])
        visited.add(self.source)

        # Perform BFS on the residual graph to find all reachable nodes from the source
        while queue:
            u = queue.popleft()
            for edge in self.graph[u]:
                if edge.residual_capacity() > 0 and edge.v not in visited:
                    visited.add(edge.v)
                    queue.append(edge.v)

        # Identify edges that go from visited to non-visited nodes in the original graph
        # These form the edges in the min-cut set
        min_cut_edges = []
        for u in visited:
            for edge in self.graph[u]:
                if edge.capacity > 0 and edge.v not in visited:
                    min_cut_edges.append((edge.u, edge.v, edge.capacity, edge.cost))

        return visited, min_cut_edges

    def dijkstra_max_flow(self):
        """
        Computes the maximum flow from source to sink using a modified Dijkstra algorithm,
        also known as the *Widest Path Algorithm*. This method repeatedly finds paths
        with the highest possible bottleneck capacity (widest path) and augments flow
        along them until no such path exists.

        Returns:
            total_flow (int): The total maximum flow from source to sink.
            flow_per_edge_list (List[Tuple[int, int, int]]): A list of tuples representing
                the flow on each original edge in the format (u, v, flow).
        """
        n = max(self.graph.keys(), default=0)  # Total number of nodes

        source = self.source
        sink = self.sink
        total_flow = 0  # Accumulator for total max flow

        def widest_path():
            """
            Finds a path from source to sink where the minimum capacity (bottleneck) 
            along the path is maximized. This is achieved using a modified version of 
            Dijkstras algorithm with edge weights replaced by residual capacities.

            Returns:
                dist (List[int]): Maximum bottleneck capacities from source to each node.
                prev_edge (List[Edge]): For each node, stores the edge used to reach it.
            """
            dist = [0] * (n + 1)           # Maximum bottleneck capacity to each node
            prev_edge = [None] * (n + 1)   # Previous edge in the path
            visited = [False] * (n + 1)    # Mark visited nodes

            dist[source] = float('inf')    # Infinite capacity at the source
            heap = [(-dist[source], source)]  # Max-heap using negative values

            while heap:
                _, u = heapq.heappop(heap)
                if visited[u]:
                    continue
                visited[u] = True

                for edge in self.graph[u]:
                    if edge.residual_capacity() > 0:
                        v = edge.v
                        cap = min(dist[u], edge.residual_capacity())  # Bottleneck capacity
                        if dist[v] < cap:
                            dist[v] = cap
                            prev_edge[v] = edge
                            heapq.heappush(heap, (-dist[v], v))  # Push with negative value for max-heap

            return dist, prev_edge

        # Augment flow along widest paths while they exist
        while True:
            dist, prev_edge = widest_path()
            if prev_edge[sink] is None:
                break  # No augmenting path found

            bottleneck = dist[sink]
            v = sink
            # Traverse the path backwards and update flows
            while v != source:
                edge = prev_edge[v]
                edge.flow += bottleneck            # Push flow forward
                edge.rev.flow -= bottleneck        # Push reverse flow backward
                v = edge.u

            total_flow += bottleneck  # Add to total flow

        # Collect final flow values for original (non-artificial) edges
        flow_per_edge_list = []
        for edge in self.edges:
            if not (edge.u == self.sink and edge.v == self.source):  # Skip artificial reverse edge if any
                flow_per_edge_list.append((edge.u, edge.v, edge.flow))

        print(f"Max flow: {total_flow}")
        return total_flow, flow_per_edge_list

    def min_cost_max_flow(self):
        """
        Computes the minimum-cost maximum-flow using the **Successive Shortest Augmenting Path** (SSAP) algorithm.
        This implementation uses Dijkstras algorithm with reduced costs to find the shortest augmenting paths,
        and node potentials to ensure non-negative edge weights.

        Returns:
            total_flow (int): The total maximum flow from source to sink.
            total_cost (int): The total cost associated with the maximum flow.
            flow_per_edge (List[Tuple[int, int, int]]): A list of flow values for each original edge
                in the format (u, v, flow).
        """
        n = max(self.graph.keys(), default=0)
        if isinstance(n, str):
            n = self.original_nodes + 50  # For cases with string-based node labels

        source = self.source
        sink = self.sink
        potential = [0] * (n + 10)  # Node potentials for reduced costs
        total_flow = 0
        total_cost = 0
        iteration = 0  # For visualization/tracing purposes

        def dijkstra():
            """
            Modified Dijkstras algorithm to find shortest augmenting path using reduced costs:
            reduced_cost = edge.cost + potential[u] - potential[v].

            Returns:
                dist (List[float]): Shortest distances from source to all nodes under reduced costs.
                prev_edge (List[Edge]): Stores the edge used to reach each node on the shortest path.
            """
            dist = [float('inf')] * (n + 10)
            prev_edge = [None] * (n + 10)
            visited = [False] * (n + 10)
            dist[source] = 0
            heap = [(0, source)]

            while heap:
                d, u = heapq.heappop(heap)
                if visited[u]:
                    continue
                visited[u] = True

                for edge in self.graph[u]:
                    if edge.residual_capacity() > 0:
                        v = edge.v
                        cost = edge.cost + potential[u] - potential[v]  # Reduced cost
                        if dist[v] > dist[u] + cost:
                            dist[v] = dist[u] + cost
                            prev_edge[v] = edge
                            heapq.heappush(heap, (dist[v], v))

            return dist, prev_edge

        # Main loop: augment flow along successive shortest paths
        while True:
            dist, prev_edge = dijkstra()

            if self.show_iterations:
                # Optionally visualize the state after each iteration
                dot_name = f"Dotfiles/graph_iter_{iteration}.dot"
                pdf_name = f"Graphs/graph_iter_{iteration}.pdf"
                self.generate_dot_and_pdf(dot_name, pdf_name, use_reduced_costs=True, potential=potential)
                iteration += 1

            if prev_edge[sink] is None:
                break  # No more augmenting paths

            # Determine bottleneck capacity of the found path
            bottleneck = float('inf')
            v = sink
            while v != source:
                edge = prev_edge[v]
                bottleneck = min(bottleneck, edge.residual_capacity())
                v = edge.u

            # Augment flow along the path and calculate total cost
            v = sink
            while v != source:
                edge = prev_edge[v]
                edge.flow += bottleneck
                edge.rev.flow -= bottleneck
                total_cost += bottleneck * edge.cost  # Actual cost, not reduced cost
                v = edge.u

            total_flow += bottleneck

            # Update node potentials to preserve reduced cost correctness
            for i in range(len(potential)):
                if dist[i] < float('inf'):
                    potential[i] += dist[i]

        print(f"Max flow: {total_flow}")
        print(f"Min cost: {total_cost}")

        # Return flow values on original edges (excluding artificial ones)
        flow_per_edge = []
        for edge in self.edges:
            if not (edge.u == self.sink and edge.v == self.source):
                flow_per_edge.append((edge.u, edge.v, edge.flow))

        return total_flow, total_cost, flow_per_edge

    def generate_dot_and_pdf(self, dot_filename="graph.dot", pdf_filename="graph.pdf", 
                             min_cut_set=None, use_reduced_costs=False, potential=None):
        """
        Generates a DOT representation of the current residual graph and converts it into a PDF
        using Graphviz. Useful for visualizing flow networks during algorithm execution.

        Args:
            dot_filename (str): Path to the output DOT file.
            pdf_filename (str): Path to the output PDF file generated from the DOT.
            min_cut_set (Optional[Set[int]]): Optional set of nodes on one side of a min-cut.
                                              Nodes not in this set can be highlighted.
            use_reduced_costs (bool): If True, uses reduced costs (based on potentials) in labels.
            potential (Optional[List[int]]): List of node potentials, used when computing reduced costs.
        """
        with open(dot_filename, "w") as f:
            f.write("digraph G {\n")
            f.write("  rankdir=LR;\n")  # Left to right layout
            f.write("  splines=polyline;\n")
            f.write("  nodesep=1.0;\n")
            f.write("  ranksep=1.0;\n")
            f.write("  node [shape=circle, width=0.5, height=0.5, fixedsize=true];\n")
            f.write(f"  {{ rank=source; {self.source}; }}\n")
            f.write(f"  {{ rank=sink; {self.sink}; }}\n")

            drawn = set()

            for u in sorted(self.graph):
                for edge in self.graph[u]:
                    key = (edge.u, edge.v)
                    rev_key = (edge.v, edge.u)
                    residual_cap = edge.residual_capacity()

                    # Draw forward edge if it has residual capacity
                    if residual_cap > 0 and key not in drawn:
                        if use_reduced_costs and potential:
                            reduced_cost = potential[edge.u] + edge.cost - potential[edge.v]
                            label = f"cap: {residual_cap}, cᵣ: {reduced_cost}"
                        else:
                            label = f"cap: {residual_cap}, cost: {edge.cost}"
                        f.write(f'  "{edge.u}" -> "{edge.v}" [label="{label}", color=black];\n')
                        drawn.add(key)

                    # Draw reverse edge if flow > 0
                    if edge.flow > 0 and rev_key not in drawn:
                        if use_reduced_costs and potential:
                            reduced_cost = -(potential[edge.u] + edge.cost - potential[edge.v])
                            label = f"cap: {edge.flow}, cᵣ: {reduced_cost}"
                        else:
                            label = f"cap: {edge.flow}, cost: {-edge.cost}"
                        f.write(f'  "{edge.v}" -> "{edge.u}" [label="{label}", color=blue, style=dashed];\n')
                        drawn.add(rev_key)

            # Highlight nodes based on cut or other heuristics
            all_nodes = set(self.graph.keys())
            for edges in self.graph.values():
                for edge in edges:
                    all_nodes.add(edge.v)

            for node in all_nodes:
                if min_cut_set and node not in min_cut_set:
                    f.write(f'  "{node}" [style=filled, fillcolor=lightcoral];\n')
                elif isinstance(node, str) and node.isdigit() and len(node) == 2:
                    f.write(f'  "{node}" [style=filled, fillcolor=lightblue];\n')

            f.write("}\n")

        try:
            subprocess.run(["dot", "-Tpdf", dot_filename, "-o", pdf_filename], check=True)
            print(f"Saved graph to {pdf_filename}")
        except Exception as e:
            print(f"Error generating graph: {e}")

    def reset_network(self):
        """
        Resets the graph to its original state by reloading it from the input file.
        Clears all edges, resets node counters, and reconstructs the graph.

        Useful when running multiple algorithms or tests on the same base network.
        """
        self.graph.clear()                         # Clear adjacency list
        self.edges.clear()                         # Clear edge objects
        self.num_nodes = self.original_nodes       # Reset number of nodes
        self.extra_node_id = self.original_nodes   # Reset ID counter for any auxiliary nodes
        self.read_graph(self.input_file)           # Rebuild graph from input


# Main routine to demonstrate usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    print(f"Loading network from: {input_file}")

    # Load and visualize initial state
    network = FlowNetwork(input_file, show_itreations=True)
    network.generate_dot_and_pdf("Dotfiles/graph_before.dot", "Graphs/graph_before.pdf")

    print("Calculate max flow")
    flow, flow_dict = network.dijkstra_max_flow()
    print("Flow per edge: ", flow_dict)

    min_cut_set, min_cut_edges = network.get_min_cut()
    print("Min cut edges: ", min_cut_edges)

    # Reset for min-cost max-flow
    network.reset_network()

    print("min cost max flow calculations: ")
    flow, cost, flow_per_edge = network.min_cost_max_flow()
    print("Flow per edge: ", flow_per_edge)

    min_cut_set, min_cut_edges = network.get_min_cut()
    print("Min cut edges: ", min_cut_edges)

    network.generate_dot_and_pdf("Dotfiles/graph_after.dot", "Graphs/graph_after.pdf", min_cut_set=min_cut_set)
