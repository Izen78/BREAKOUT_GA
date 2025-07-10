from enum import Enum

# Call mutation with 0.01% chance
MUTATION_RATE = 0.01

class NodeGene:
    class NodeType(Enum):
        Sensor = 0
        Hidden = 1
        Output = 2

    node_type = None
    node_id = None

    def __init__(self, node_type, node_id) -> None:
        self.node_type = node_type
        self.node_id = node_id

    def get_type(self):
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


class Genome:
    node_gene = []
    connection_gene = []

    def __init__(self) -> None:
        pass


    def h_connection_exists(self):
        pass

    # go through all unconnected nodes as potential new connection genes and choose one at random
    def add_connection_mutation(self):
        # make a list of all available connections
        for node1 in self.node_gene:
            for node2 in self.node_gene:
                pass


