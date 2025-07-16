from neat import *


population = []
innovation = InnovationTracker()
innovation.set_innovation(13)
# Initial Input Nodes
input_node1 = NodeGene(NodeType.Sensor, 1) # x position of paddle 
input_node2 = NodeGene(NodeType.Sensor, 2) # x position of ball
input_node3 = NodeGene(NodeType.Sensor, 3) # y position of ball 
input_node4 = NodeGene(NodeType.Sensor, 4) # ball.angle

# Initial Output Nodes
output_node1 = NodeGene(NodeType.Output, 5) # Travel left?
output_node2 = NodeGene(NodeType.Output, 6) # Travel right?
output_node3 = NodeGene(NodeType.Output, 7) # Do nothing?
for i in range(10):
    initial_genome = Genome()

    initial_genome.add_node_gene(input_node1.copy())
    initial_genome.add_node_gene(input_node2.copy())
    initial_genome.add_node_gene(input_node3.copy())
    initial_genome.add_node_gene(input_node4.copy())
    initial_genome.add_node_gene(output_node1.copy())
    initial_genome.add_node_gene(output_node2.copy())
    initial_genome.add_node_gene(output_node3.copy())
    # Connections from input node 1
    connection_11 = ConnectionGene(input_node1.get_id(), output_node1.get_id(), random.random()*5, True, 1)
    connection_12 = ConnectionGene(input_node1.get_id(), output_node2.get_id(), random.random()*5, True, 2)
    connection_13 = ConnectionGene(input_node1.get_id(), output_node3.get_id(), random.random()*5, True, 3)
    # Connections from input node 2
    connection_21 = ConnectionGene(input_node2.get_id(), output_node1.get_id(), random.random()*5, True, 4)
    connection_22 = ConnectionGene(input_node2.get_id(), output_node2.get_id(), random.random()*5, True, 5)
    connection_23 = ConnectionGene(input_node2.get_id(), output_node3.get_id(), random.random()*5, True, 6)
    # Connections from input node 3
    connection_31 = ConnectionGene(input_node3.get_id(), output_node1.get_id(), random.random()*5, True, 7)
    connection_32 = ConnectionGene(input_node3.get_id(), output_node2.get_id(), random.random()*5, True, 8)
    connection_33 = ConnectionGene(input_node3.get_id(), output_node3.get_id(), random.random()*5, True, 9)
    # Connections from input node 4
    connection_41 = ConnectionGene(input_node4.get_id(), output_node1.get_id(), random.random()*5, True, 10)
    connection_42 = ConnectionGene(input_node4.get_id(), output_node2.get_id(), random.random()*5, True, 11)
    connection_43 = ConnectionGene(input_node4.get_id(), output_node3.get_id(), random.random()*5, True, 12)

    initial_genome.add_connection_gene(connection_11)
    initial_genome.add_connection_gene(connection_12)
    initial_genome.add_connection_gene(connection_13)
    initial_genome.add_connection_gene(connection_21)
    initial_genome.add_connection_gene(connection_22)
    initial_genome.add_connection_gene(connection_23)
    initial_genome.add_connection_gene(connection_31)
    initial_genome.add_connection_gene(connection_32)
    initial_genome.add_connection_gene(connection_33)
    initial_genome.add_connection_gene(connection_41)
    initial_genome.add_connection_gene(connection_42)
    initial_genome.add_connection_gene(connection_43)

    population.append(initial_genome)

for i in range(100):
    elem1 = random.choice(population).clone()
    elem2 = random.choice(population).clone()
    print("PARENT 1")
    elem1.print_genome()
    print("PARENT 2")
    elem2.print_genome()

    print(f"Test {i}")
    child = Genome.crossover(elem1, elem2, False)

    child.print_genome()

    child.add_node_mutation(innovation)


    # child.add_connection_mutation(innovation)

    child.print_genome()
