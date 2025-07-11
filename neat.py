from __future__ import annotations
from enum import Enum
import random
from typing import Type


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
    
class ConnectionGene:
    in_node = None
    out_node = None
    weight = 0
    expressed = False
    innovation_number = 0

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


class Genome:

    def __init__(self) -> None:
        self.node_gene = []
        self.connection_gene = []



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
    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        pass
