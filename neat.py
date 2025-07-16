# TODO: Hashing
from __future__ import annotations
from enum import Enum
import random
from typing import Type
import networkx as nx
import matplotlib.pyplot as plt


# Call mutation with 0.01% chance
MUTATION_RATE = 0.01

class NodeType(Enum):
    Sensor = 0
    Hidden = 1 
    Output = 2

class InnovationTracker:
    def __init__(self) -> None:
        self.current_innovation = 0
        self.connection_hist = {}
        self.node_history = {}
        self.next_node_id = 8

    def increment(self):
        self.current_innovation += 1

    def get_innovation(self) -> int:
        return self.current_innovation

    def set_innovation(self, new_inn) -> None:
        self.current_innovation = new_inn

    def get_or_create_connection(self, in_node: int, out_node: int) -> int:
        key = (in_node, out_node)
        if key not in self.connection_hist:
            self.connection_hist[key] = self.current_innovation
            self.current_innovation += 1
        return self.connection_hist[key]

    def get_or_create_node_split(self, in_node: int, out_node: int) -> tuple[int, int, int]:
        key = (in_node, out_node)
        if key not in self.node_history:
            node_id = self.next_node_id
            self.next_node_id += 1
            innov1 = self.current_innovation
            innov2 = self.current_innovation + 1
            self.current_innovation += 2
            self.node_history[key] = (node_id, innov1, innov2)
        return self.node_history[key]


class NodeGene:

    node_type = None
    node_id = None

    def __init__(self, node_type: NodeType, node_id: int) -> None:
        self.node_type = node_type
        self.node_id = node_id

    def get_type(self) -> NodeType:
        return self.node_type
    
    def get_id(self):
        return self.node_id

    def copy(self) -> NodeGene:
        return NodeGene(self.node_type, self.node_id)

    def __str__(self) -> str:
        return str(self.node_id) + ";" + str(self.node_type)

    def copy(self):
        return NodeGene(self.node_type, self.node_id)
    
class ConnectionGene:

    def __init__(self, in_node, out_node, weight, expressed, inn_number) -> None:
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.expressed = expressed
        self.innovation_number = inn_number

    def get_in_node(self):
        return self.in_node

    def get_out_node(self):
        return self.out_node
    
    def get_weight(self):
        return self.weight

    def is_expressed(self):
        return self.expressed

    def get_innov(self):
        return self.innovation_number

    def __str__(self) -> str:
        return "in: " + str(self.in_node) + ", out: " + str(self.out_node) + ", weight: " + str(self.weight) + ", inn: " + str(self.innovation_number) + ", exp?: " + str(self.expressed)

    def copy(self):
        return ConnectionGene(self.in_node, self.out_node, self.weight, self.expressed, self.innovation_number)


class Genome:

    def __init__(self) -> None:
        self.node_gene = []
        self.connection_gene = []

    def get_node_genes(self) -> list[NodeGene]:
        return self.node_gene

    def get_connection_genes(self) -> list[ConnectionGene]:
        return self.connection_gene

    def add_node_gene(self, node) -> None:
        self.node_gene.append(node)

    def add_connection_gene(self, conn) -> None:
        self.connection_gene.append(conn)

    def search_gene(self, inn_index) -> ConnectionGene | None:
        for conn in self.connection_gene:
            if conn.get_innov() == inn_index:
                return conn
        return None
    
    def print_genome(self) -> None:
        edges = []
        for conn in self.connection_gene:
            if conn.is_expressed():
                edges.append([conn.get_in_node(), conn.get_out_node()])
        graph = nx.Graph()
        graph.add_edges_from(edges)
        nx.draw_networkx(graph)
        plt.show()

    # go through all unconnected nodes as potential new connection genes and choose one at random
    # def add_connection_mutation(self, innovation_tracker: InnovationTracker):
    #     reversed = False
    #     node1 = self.node_gene[random.randint(0, len(self.node_gene)-1)]
    #     node2 = self.node_gene[random.randint(0, len(self.node_gene)-1)]
    #     mutated_weight = random.uniform(0.0, 1.0)#*2-1
    #
    #     if node1.get_type() == NodeType.Output and node2.get_type() == NodeType.Hidden:
    #         reversed = True
    #     elif node1.get_type() == NodeType.Hidden and node2.get_type() == NodeType.Sensor:
    #         reversed = True
    #     elif node1.get_type() == NodeType.Output and node2.get_type() == NodeType.Sensor:
    #         reversed = True
    #
    #     # check for existing gene connections
    #     existing = False
    #     for con in self.connection_gene:
    #         if con.in_node == node1.get_id() and con.out_node == node2.get_id() or con.in_node == node2.get_id() and con.out_node == node1.get_id():
    #             existing = True
    #             break
    #
    #     if not existing:
    #         mutated_connection = ConnectionGene(node2.get_id() if reversed else node1.get_id(), node1.get_id() if reversed else node2.get_id(), mutated_weight, True, innovation_tracker.get_innovation())
    #         self.connection_gene.append(mutated_connection)

    def add_connection_mutation(self, innovation_tracker):
        # Try a reasonable number of times to find a valid connection
        for _ in range(100):
            node1 = random.choice(self.node_gene)
            node2 = random.choice(self.node_gene)

            # Prevent connections from output to input
            if node1.get_type() == NodeType.Output or node2.get_type() == NodeType.Sensor:
                continue

            # Prevent self-connection
            if node1.get_id() == node2.get_id():
                continue

            # Prevent duplicates
            already_connected = any(
                conn.get_in_node() == node1.get_id() and conn.get_out_node() == node2.get_id()
                for conn in self.connection_gene
            )
            if already_connected:
                continue

            # Prevent cycles
            if creates_cycle(node1.get_id(), node2.get_id(), self.connection_gene):
                continue

            # Passed all checks → create connection
            innov = innovation_tracker.get_or_create_connection(node1.get_id(), node2.get_id())
            new_conn = ConnectionGene(
                node1.get_id(),
                node2.get_id(),
                random.uniform(-1, 1),
                True,
                innov
            )
            self.connection_gene.append(new_conn)
            print(f"adding connection: {str(new_conn)}")
            return  # success

        # Failed to find valid connection
        print("Failed to add connection: no valid pair found.")


    # def add_node_mutation(self, innovation_tracker: InnovationTracker):
    #     # ADD functionality here
    #
    #     node_id, inn1, inn2 = innovation_tracker.get_or_create_node_split(in_node, out_node)
    #     middle_node = NodeGene(NodeType.Hidden, node_id)
    #     # middle_node = NodeGene(NodeType.Hidden, max(ids)+1)
    #     connection1 = ConnectionGene(in_node, node_id, 1.0, True, inn1)
    #     connection2 = ConnectionGene(node_id, out_node, connection.get_weight(), True, inn2)
    #
    #     self.node_gene.append(middle_node)
    #     self.connection_gene.append(connection1)
    #     self.connection_gene.append(connection2)
        
    def add_node_mutation(self, innovation_tracker: InnovationTracker):
        # Choose a random expressed connection to split
        expressed_connections = [c for c in self.connection_gene if c.is_expressed()]
        if not expressed_connections:
            return

        connection = random.choice(expressed_connections)
        in_node = connection.get_in_node()
        out_node = connection.get_out_node()

        # Disable the original connection
        connection.expressed = False

        # Get or create the new node and connection innovations
        node_id, inn1, inn2 = innovation_tracker.get_or_create_node_split(in_node, out_node)

        # Avoid duplicating the node
        if all(n.get_id() != node_id for n in self.node_gene):
            middle_node = NodeGene(NodeType.Hidden, node_id)
            self.node_gene.append(middle_node)

        # Add the two new connections
        conn1 = ConnectionGene(in_node, node_id, 1.0, True, inn1)
        conn2 = ConnectionGene(node_id, out_node, connection.get_weight(), True, inn2)
        self.connection_gene.append(conn1)
        self.connection_gene.append(conn2)

        print(f"Added node mutation: split {in_node} → {out_node} into {in_node} → {node_id} → {out_node}")

    # Parent1 is the fitter parent
    # @staticmethod
    # def crossover(parent1: Genome, parent2: Genome, equal_fitness: bool) -> Genome:
    #     child = Genome()
    #
    #     for conn1 in parent1.get_connection_genes():
    #         conn2 = parent2.search_gene(conn1.get_innov())
    #         if conn2 is not None:
    #             # there exists overlapping connections
    #             # pick between 2 connections randomly
    #             x = random.random()
    #             child.add_connection_gene(conn1 if x >=0.5 else conn2)
    #         else:
    #             # it is either disjoint/excess
    #             # TODO: if equal fitness there should be a chance of getting disjoint/excess genes
    #             child.add_connection_gene(conn1)
    #
    #     if equal_fitness:
    #         for conn2 in parent2.get_connection_genes():
    #             if child.search_gene(conn2.get_innov()) == None:
    #                 x = random.random()
    #                 if x >= 0.5:
    #                     child.add_connection_gene(conn2)
    #
    #
    #     # add the nodes needed if not already inside
    #     existing_ids = {node.get_id(): node for node in parent1.get_node_genes() + parent2.get_node_genes()}
    #
    #     for output_node in parent1.get_node_genes() + parent2.get_node_genes():
    #         if output_node.get_type() in (NodeType.Output, NodeType.Sensor):
    #             existing_ids[output_node.get_id()] = output_node
    #     for ch_conn in child.get_connection_genes():
    #         conn_in_id = ch_conn.get_in_node()
    #         conn_out_id = ch_conn.get_out_node()
    #         if conn_in_id not in [n.get_id() for n in child.get_node_genes()]:
    #             child.add_node_gene(existing_ids[conn_in_id])
    #         if conn_out_id not in [n.get_id() for n in child.get_node_genes()]:
    #             child.add_node_gene(existing_ids[conn_out_id])
    #
    #     output_ids = [node.get_id() for node in parent1.get_node_genes() if node.get_type() == NodeType.Output or node.get_type() == NodeType.Sensor]
    #     for node_id in output_ids:
    #         if node_id not in [n.get_id() for n in child.get_node_genes()]:
    #             if node_id in existing_ids:
    #                 child.add_node_gene(existing_ids[node_id])
    #             else:
    #                 print(f"[ERROR] Output node {node_id} missing in both parents")
    #
    #     return child


    # @staticmethod
    # def crossover(parent1: Genome, parent2: Genome, equal_fitness: bool) -> Genome:
    #     child = Genome()
    #
    #     # Cross over functionality
    #
    #     return child
    
    @staticmethod
    def crossover(parent1: Genome, parent2: Genome, equal_fitness: bool) -> Genome:
        child = Genome()

        # Map connections by innovation number
        conn1_map = {conn.get_innov(): conn for conn in parent1.get_connection_genes()}
        conn2_map = {conn.get_innov(): conn for conn in parent2.get_connection_genes()}

        all_innovs = set(conn1_map.keys()).union(set(conn2_map.keys()))

        for innov in sorted(all_innovs):
            conn1 = conn1_map.get(innov)
            conn2 = conn2_map.get(innov)

            if conn1 and conn2:
                # Matching gene → pick randomly from either parent
                chosen = conn1 if random.random() < 0.5 else conn2
                child.add_connection_gene(chosen.copy())
            elif conn1 and not conn2:
                # Disjoint or excess gene from parent1
                if not equal_fitness:
                    child.add_connection_gene(conn1.copy())
                else:
                    if random.random() < 0.5:
                        child.add_connection_gene(conn1.copy())
            elif conn2 and not conn1:
                # Disjoint or excess gene from parent2
                if equal_fitness:
                    if random.random() < 0.5:
                        child.add_connection_gene(conn2.copy())
                # else skip (not fitter)

        # Add necessary nodes from both parents
        existing_nodes = {node.get_id(): node for node in parent1.get_node_genes() + parent2.get_node_genes()}
        used_node_ids = set()
        
        for node in parent1.get_node_genes() + parent2.get_node_genes():
            if node.get_type() in (NodeType.Sensor, NodeType.Output):
                used_node_ids.add(node.get_id())
                child.add_node_gene(node)

        for conn in child.get_connection_genes():
            used_node_ids.add(conn.get_in_node())
            used_node_ids.add(conn.get_out_node())

        for node_id in used_node_ids:
            if node_id in existing_nodes:
                child.add_node_gene(existing_nodes[node_id])
            else:
                print(f"[WARNING] Missing node ID {node_id} in both parents")

        return child
    
    def clone(self) -> Genome:
        new_genome = Genome()
        new_genome.node_gene = [node.copy() for node in self.node_gene]
        new_genome.connection_gene = [conn.copy() for conn in self.connection_gene]
        
        return new_genome

class Population:
    pass 

class NeuronInput:
    def __init__(self, input_id, weight) -> None:
        self.input_id = input_id;
        self.weight = weight

class Neuron:
    def __init__(self, activation_fn, bias, inputs: list[NeuronInput]):
        self.activation_fn = activation_fn
        self.bias = bias
        self.inputs = inputs

class FeedForwardNeuralNetwork:

    def __init__(self) -> None:
        self.layers = []
        self.nodes = []
        self.connections = []
        self.inputs = []
        self.outputs = []

    def phenotype_from(self, genome: Genome) -> FeedForwardNeuralNetwork:
        self.nodes = genome.get_node_genes()
        activated = []
        # Get rid of disable connections
        for node in self.nodes:
            if node.get_type() == NodeType.Sensor:
                self.inputs.append(node) 
                activated.append(node)
            elif node.get_type() == NodeType.Output:
                self.outputs.append(node)

        for connection in genome.get_connection_genes():
            if connection.is_expressed():
                self.connections.append(connection)

        # Categorise nodes into layers
        # Add all sensors as activated initially

        self.layers.append(activated[:])
        while len(activated) < len(self.nodes):
            temp = []
            for node in self.nodes:
                if node not in activated:
                    all_inputs_activated = True
                    for conn in self.connections:
                        if conn.get_out_node() == node.get_id():
                            in_node_id = conn.get_in_node()
                            in_node = next((n for n in self.nodes if n.get_id() == in_node_id), None)
                            if in_node not in activated:
                                all_inputs_activated = False
                                break
                    if all_inputs_activated:
                        temp.append(node)

            if not temp:
                print("cannot activate any more nodes")
                break
            self.layers.append(temp)
            activated.extend(temp)

        assert len(self.inputs) > 0, "ERROR: NO INPUT NODES IN GENOME"
        return self

    # sets node output values after feed forward process 
    def activate(self, inputs: dict[GeneNode,float], activation_fn) -> dict[int, float]:
        values = {} # node_id : value
        # for output_node in self.outputs:
        #     values[output_node] = 0.0

        # set up initial input values
        for node,val in inputs.items():
            values[node.get_id()] = val

        for layer in self.layers:
            for current_node in layer:
                input_value = 0
                for conn in self.connections:
                    if conn.get_out_node() == current_node.get_id():
                        input_value += conn.get_weight() * values.get(conn.get_in_node(), 0.0)
                values[current_node.get_id()] = activation_fn(input_value)
        return values


def creates_cycle(in_node_id, out_node_id, connections):
    visited = set()
    stack = [out_node_id]

    while stack:
        current = stack.pop()
        if current == in_node_id:
            return True  # cycle detected
        for conn in connections:
            if conn.is_expressed() and conn.get_in_node() == current:
                stack.append(conn.get_out_node())
    return False

# if __name__ == "__main__":
    # test1 = Genome()
    # test2 = Genome()
    #
    # node1 = NodeGene(NodeType.Sensor, 1)
    # node2 = NodeGene(NodeType.Sensor, 2)
    # node3 = NodeGene(NodeType.Sensor, 3)
    # node4 = NodeGene(NodeType.Output, 4)
    # node5 = NodeGene(NodeType.Hidden, 5)
    # node6 = NodeGene(NodeType.Hidden, 6)
    #
    # conn1 = ConnectionGene(node1, node4, 0.5, True, 1)
    # conn2 = ConnectionGene(node2, node4, 0.5, False, 2)
    # conn3 = ConnectionGene(node3, node4, 0.5, True, 3)
    # conn4 = ConnectionGene(node2, node5, 0.5, True, 4)
    #
    # conn5 = ConnectionGene(node5, node4, 0.5, True, 5)
    # conn5_2 = ConnectionGene(node5, node4, 0.5, False, 5)
    #
    # conn6 = ConnectionGene(node5, node6, 0.5, True, 6)
    # conn7 = ConnectionGene(node6, node4, 0.5, True, 7)
    # conn9 = ConnectionGene(node3, node5, 0.5, True, 9)
    # conn10 = ConnectionGene(node1, node6, 0.5, True, 10)
    #
    # conn8 = ConnectionGene(node1, node5, 0.5, True, 8)
    #
    # test1.add_connection_gene(conn1)
    # test1.add_connection_gene(conn2)
    # test1.add_connection_gene(conn3)
    # test1.add_connection_gene(conn4)
    # test1.add_connection_gene(conn5)
    # test1.add_connection_gene(conn8)
    # 
    # test2.add_connection_gene(conn1)
    # test2.add_connection_gene(conn2)
    # test2.add_connection_gene(conn3)
    # test2.add_connection_gene(conn4)
    # test2.add_connection_gene(conn5_2)
    # test2.add_connection_gene(conn6)
    # test2.add_connection_gene(conn7)
    # test2.add_connection_gene(conn9)
    # test2.add_connection_gene(conn10)
    #
    # child = Genome.crossover(test1, test2, True)
    # child.print_genome()
    #
