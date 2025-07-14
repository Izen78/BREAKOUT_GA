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

    def increment(self):
        self.current_innovation += 1

    def get_innovation(self) -> int:
        return self.current_innovation

    def set_innovation(self, new_inn) -> None:
        self.current_innovation = new_inn

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
        return str(self.node_id)
    
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
        return "in: " + str(self.in_node) + ", out: " + str(self.out_node) + ", inn: " + str(self.innovation_number) + ", exp?: " + str(self.expressed)


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
    def add_connection_mutation(self, innovation_tracker: InnovationTracker):
        reversed = False
        node1 = self.node_gene[random.randint(0, len(self.node_gene)-1)]
        node2 = self.node_gene[random.randint(0, len(self.node_gene)-1)]
        mutated_weight = random.uniform(0.0, 1.0)#*2-1

        if node1.get_type() == NodeType.Output and node2.get_type() == NodeType.Hidden:
            reversed = True
        elif node1.get_type() == NodeType.Hidden and node2.get_type() == NodeType.Input:
            reversed = True
        elif node1.get_type() == NodeType.Output and node2.get_type() == NodeType.Input:
            reversed = True

        # check for existing gene connections
        existing = False
        for con in self.connection_gene:
            if con.in_node == node1.get_id() and con.out_node == node2.get_id() or con.in_node == node2.get_id() and con.out_node == node1.get_id():
                existing = True
                break

        if not existing:
            mutated_connection = ConnectionGene(node2.get_id() if reversed else node1.get_id(), node1.get_id() if reversed else node2.get_id(), mutated_weight, True, innovation_tracker.get_innovation())
            self.connection_gene.append(mutated_connection)


    def add_node_mutation(self, innovation_tracker: InnovationTracker):
        # get random existing connection
        connection = self.connection_gene[random.randint(0, len(self.connection_gene)-1)]

        in_node = connection.get_in_node()
        out_node = connection.get_out_node()

        # disable existing connection
        connection.expressed = False

        # make new middle node
        ids = []
        for node in self.node_gene:
            ids.append(node.get_id())

        middle_node = NodeGene(NodeType.Hidden, max(ids)+1)
        connection1 = ConnectionGene(in_node.get_id(), middle_node.get_id(), 1.0, True, innovation_tracker.get_innovation())
        connection2 = ConnectionGene(middle_node.get_id(), out_node.get_id(), connection.get_weight(), True, innovation_tracker.get_innovation())

        self.node_gene.append(middle_node)
        self.connection_gene.append(connection1)
        self.connection_gene.append(connection2)
        

    # Parent1 is the fitter parent
    @staticmethod
    def crossover(parent1: Genome, parent2: Genome, equal_fitness: bool) -> Genome:
        child = Genome()

        for conn1 in parent1.get_connection_genes():
            conn2 = parent2.search_gene(conn1.get_innov())
            if conn2 is not None:
                # there exists overlapping connections
                # pick between 2 connections randomly
                x = random.random()
                child.add_connection_gene(conn1 if x >=0.5 else conn2)
            else:
                # it is either disjoint/excess
                # TODO: if equal fitness there should be a chance of getting disjoint/excess genes
                child.add_connection_gene(conn1)

        if equal_fitness:
            for conn2 in parent2.get_connection_genes():
                if child.search_gene(conn2.get_innov()) == None:
                    x = random.random()
                    if x >= 0.5:
                        child.add_connection_gene(conn2)

        # add the nodes needed if not already inside
        for ch_conn in child.get_connection_genes():
            conn_in = ch_conn.get_in_node()
            conn_out = ch_conn.get_out_node()
            if conn_in not in child.get_node_genes():
                child.add_node_gene(conn_in)
            elif conn_out not in child.get_node_genes():
                child.add_node_gene(conn_out)

        return child

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

    def phenotype_from(self, genome: Genome) -> None:
        self.nodes = genome.get_node_genes()
        # Get rid of disable connections
        for node in self.nodes:
            if node.get_type() == NodeType.Sensor:
                self.inputs.append(node) 
            elif node.get_type() == NodeType.Output:
                self.outputs.append(node)

        for connection in genome.get_connection_genes():
            if connection.is_expressed():
                self.connections.append(connection)

        # Categorise nodes into layers
        activated = []
        # Add all sensors as activated initially
        for conn in self.connections:
            if conn.get_in_node() == NodeType.Sensor:
                activated.append(conn)

        self.layers.append(activated)
        while len(activated) != len(self.nodes):
            temp = []
            for node in self.nodes:
                if node not in activated:
                    i = 0
                    while i < len(self.connections):
                        if node.get_id() == self.connections[i].get_out_node():
                            if self.connections[i].get_in_node() not in activated:
                                break
                        i += 1
                    if i == len(self.connections):
                        temp.append(node)
            if len(temp) > 0:
                self.layers.append(temp)
            for n in temp:
                activated.append(n)

    # sets node output values after feed forward process 
    def activate(self, inputs: dict[GeneNode,float], activation_fn) -> dict[int, float]:
        values = {} # node_id : value
        # for output_node in self.outputs:
        #     values[output_node] = 0.0

        # set up initial input values
        for i in inputs:
            values[i] = inputs[i]

        for layer in self.layers:
            for current_node in layer:
                input_value = 0
                for conn in self.connections:
                    if conn.get_out_node() == current_node.get_id():
                        in_node = conn.get_in_node()
                        input_value += in_node.get_weight() * values[in_node.get_id()]
                values[current_node.get_id()] = activation_fn(input_value)
        return values

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
