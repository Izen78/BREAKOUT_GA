# TODO:
# Add in angle threshold
# Fixed?: Fix for VEL=5, by making sure you take into account image_width and height
# make ball directions into enum
# fix ball angle with paddle

import math
import pygame
import os
from typing import TYPE_CHECKING,  Type
import random
from enum import Enum

from neat import *

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 400
TOPLEFT_CORNER = (33, 60)
BRICK_TOPLEFT_CORNER= (0, -10)

class Direc(Enum):
    NULL = -1 
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

# Import Images
i_bg = pygame.image.load("imgs/bg.png")
i_one = pygame.image.load("imgs/one.png")
i_rb = pygame.image.load("imgs/hot_dog.png")
i_ob = pygame.image.load("imgs/donut.png")
i_br_b = pygame.image.load("imgs/cookie.png")
i_yb = pygame.image.load("imgs/burrito.png")
i_gb = pygame.image.load("imgs/jelly.png")
i_bb = pygame.image.load("imgs/swiss_roll.png")
i_ball = pygame.image.load("imgs/ball.png")
i_paddle = pygame.image.load("imgs/bar.png")

class Ball:
    angle = 0 # radians, up is [0, pi] and down is (pi, 2pi)
    VEL = 2 
    img = i_ball
    y_orientation = 1
    x_orientation = 0
    ANGLE_THRESHOLD = math.pi/8
    alive = True

    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.angle =  random.uniform(math.pi, 2*math.pi)

    def move(self) -> None:
        self.x += self.VEL * math.cos(self.angle)
        self.y -= self.VEL * math.sin(self.angle)

    def reflect(self, direc) -> None:
        if direc == "left":
            if self.angle <= math.pi and self.angle >= math.pi/2:
                x_ang = math.pi - self.angle
                self.angle = x_ang
                self.x_orientation = 1
                self.y_orientation = 1
            elif self.angle >= math.pi and self.angle <= 3*math.pi/2:
                x_ang = self.angle - math.pi
                self.angle = 2*math.pi - x_ang
                self.x_orientation = 1
                self.y_orientation = -1
        elif direc == "right":
            if self.angle <= math.pi/2 and self.angle >= 0:
                x_ang = self.angle
                self.angle = math.pi - x_ang
                self.x_orientation = -1
                self.y_orientation = 1
            elif self.angle <= 2*math.pi and self.angle >= 3*math.pi/2:
                x_ang = 2*math.pi - self.angle
                self.angle = math.pi + x_ang
                self.y_orientation = -1
                self.x_orientation = -1
        elif direc == "up":
            if self.angle >= 0 and self.angle <= math.pi/2:
                x_ang = self.angle
                self.angle = 2*math.pi - x_ang
                self.x_orientation = 1
                self.y_orientation = -1
            elif self.angle <= math.pi and self.angle >= math.pi/2:
                x_ang = math.pi - self.angle
                self.angle = math.pi + x_ang
                self.x_orientation = -1
                self.y_orientation = -1
        elif direc == "down":
            if self.angle >= 3*math.pi/2 and self.angle <= 2*math.pi:
                x_ang = 2*math.pi - self.angle
                self.angle = x_ang
                self.x_orientation = 1
                self.y_orientation = 1
            if self.angle >= math.pi and self.angle <= 3*math.pi/2:
                x_ang = self.angle - math.pi
                self.angle = math.pi - x_ang
                self.x_orientation = -1
                self.y_orientation = 1

        if self.y_orientation == 1:
            if self.x_orientation == 1:
                if self.angle <= self.ANGLE_THRESHOLD:
                    self.angle += self.ANGLE_THRESHOLD
                elif self.angle <= math.pi/2 and self.angle >= math.pi/2-self.ANGLE_THRESHOLD:
                    self.angle -= self.ANGLE_THRESHOLD
        self.angle = self.angle % (2*math.pi)

            # BUG: LEFT OFF HERE - Need to implement angle threshold for each of the four quadrants so ball doesn't drill through multiple bricks then I can add the paddle
            
        ball.move()

class Brick:
    score_val = None
    alive = True
    img = None
    x = 0
    y = 0
    alive = True

    def __init__(self, score_val) -> None:
        self.score_val = score_val

    def collision(self, ball: Type[Ball]) -> bool:
        next_bx = int(ball.x) + ball.VEL*math.cos(ball.angle)
        next_by = int(ball.y) + ball.VEL*math.sin(ball.angle)

        if next_bx <= self.x + self.img.get_width() and next_bx >= self.x:
            # Either directly above or below brick
            if next_by <= self.y + self.img.get_height() and next_by >= self.y and ball.y_orientation == 1: # reflect from top
                # BELOW
                # ball.y = (self.y + self.img.get_height())*math.sin(ball.angle)
                ball.reflect("up")
                return True
            elif next_by+ball.img.get_height() >= self.y and next_by+ball.img.get_height() <= self.y + self.img.get_height() and ball.y_orientation == -1:
                # ABOVE
                ball.reflect("down")
                return True
        elif next_by >= self.y and next_by <= self.y + self.img.get_height():
            # Either directly left or right of brick
            if next_bx <= self.x + self.img.get_width() and next_bx >= self.x:
                # REFLECT FROM THE LEFT
                ball.reflect("left")
                return True
            elif next_bx+ball.img.get_width() >= self.x and next_bx+ball.img.get_width() <= self.x + self.img.get_width():
                # REFLECT FROM RIGHT
                ball.reflect("right")
                return True
        return False

class Red_Brick(Brick):
    score_val = 100
    img = i_rb
    
    def __init__(self) -> None:
        super().__init__(self.score_val)


class Orange_Brick(Brick):
    score_val = 75 
    img = i_ob
    
    def __init__(self) -> None:
        super().__init__(self.score_val)

class Brown_Brick(Brick):
    score_val = 50
    img = i_br_b
    
    def __init__(self) -> None:
        super().__init__(self.score_val)

class Yellow_Brick(Brick):
    score_val = 25
    img = i_yb
    
    def __init__(self) -> None:
        super().__init__(self.score_val)

class Green_Brick(Brick):
    score_val = 10
    img = i_gb
    
    def __init__(self) -> None:
        super().__init__(self.score_val)

class Blue_Brick(Brick):
    score_val = 5
    img = i_bb
    
    def __init__(self) -> None:
        super().__init__(self.score_val)


class Paddle:
    img = i_paddle
    y = WINDOW_HEIGHT*0.75
    x = WINDOW_WIDTH/2
    VEL = 5

    def __init__(self) -> None:
        pass

    # must constrain between grey boxes
    def move(self, direc: Direc) -> None:
        if direc == Direc.LEFT:
            self.x = max(self.x - self.VEL, TOPLEFT_CORNER[0])
        elif direc == Direc.RIGHT:
            self.x = min(self.x + self.VEL, WINDOW_WIDTH - TOPLEFT_CORNER[0] - self.img.get_width() + 2)

    def collision(self, ball: Type[Ball]) -> bool:
        next_bx = int(ball.x) + ball.VEL*math.cos(ball.angle)
        next_by = int(ball.y) + ball.VEL*math.sin(ball.angle)

        if next_bx <= self.x + self.img.get_width() and next_bx >= self.x:
            # Either directly above or below brick
            if next_by <= self.y + self.img.get_height() and next_by >= self.y and ball.y_orientation == 1: # reflect from top
                # BELOW
                # ball.y = (self.y + self.img.get_height())*math.sin(ball.angle)
                ball.reflect("up")
                return True
            elif next_by+ball.img.get_height() >= self.y and next_by+ball.img.get_height() <= self.y + self.img.get_height() and ball.y_orientation == -1:
                # ABOVE
                ball.reflect("down")
                return True
        elif next_by >= self.y and next_by <= self.y + self.img.get_height():
            # Either directly left or right of brick
            if next_bx <= self.x + self.img.get_width() and next_bx >= self.x:
                # REFLECT FROM THE LEFT
                ball.reflect("left")
                return True
            elif next_bx+ball.img.get_width() >= self.x and next_bx+ball.img.get_width() <= self.x + self.img.get_width():
                # REFLECT FROM RIGHT
                ball.reflect("right")
                return True
        return False

# Returns index of maximum element in list
def max_index(target_list: list[int]) -> int:
    best_i = 0
    i = 0
    max = float('-inf')
    for elem in target_list:
        if elem > max:
            max = elem
            best_i = i
        i += 1

    return best_i
            

pygame.init()
pygame.font.init() # 
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()
running = True
font = pygame.font.Font(pygame.font.get_default_font(), 30)

# Initial Screen
screen.fill((255,255,255))
i_bg.set_colorkey((0,0,0))

# 18 red bricks across (12x32) first brick starts at x=32, y=64
rbs = []
obs = []
br_bs = []
ybs = []
gbs = []
bbs = []
for i in range(18):
    rbs.append(Red_Brick())
    obs.append(Orange_Brick())
    br_bs.append(Brown_Brick())
    ybs.append(Yellow_Brick())
    gbs.append(Green_Brick())
    bbs.append(Blue_Brick())


bricks = [rbs, obs, br_bs, ybs, gbs, bbs]

ball = Ball(WINDOW_WIDTH/1.5, WINDOW_HEIGHT/1.5)
# ball = Ball(WINDOW_WIDTH/1.5, 70)
# paddle = Paddle()
delta_time = 0.1
moving_right = False
moving_left = False
score = 0
FITNESS_SCORES_THRESHOLD = 4770 + 60*5 # all blocks + 5 seconds

gen = 0
N = 50
population = []
fitness_scores = []
networks = []
brick_stats = [bricks for i in range(N)]
paddle_stats = [Paddle() for i in range(N)]
ball_stats = [Ball(WINDOW_WIDTH/1.5, WINDOW_HEIGHT/1.5) for i in range(N)]

innovation = InnovationTracker()
innovation.set_innovation(7)
# Initial Input Nodes
input_node1 = NodeGene(NodeType.Sensor, 1) # distance from paddle to ball
input_node2 = NodeGene(NodeType.Sensor, 2) # x position of paddle
# Initial Output Nodes
output_node1 = NodeGene(NodeType.Output, 3) # Travel left?
output_node2 = NodeGene(NodeType.Output, 4) # Travel right?
output_node3 = NodeGene(NodeType.Output, 5) # Do nothing?
for i in range(N):
    initial_genome = Genome()

    initial_genome.add_node_gene(input_node1)
    initial_genome.add_node_gene(input_node2)
    initial_genome.add_node_gene(output_node1)
    initial_genome.add_node_gene(output_node2)
    initial_genome.add_node_gene(output_node3)
    # Connections from input node 1
    connection_11 = ConnectionGene(input_node1.get_id(), output_node1.get_id(), random.random()*5, True, 1)
    connection_12 = ConnectionGene(input_node1.get_id(), output_node2.get_id(), random.random()*5, True, 2)
    connection_13 = ConnectionGene(input_node1.get_id(), output_node3.get_id(), random.random()*5, True, 3)
    # Connections from input node 2
    connection_21 = ConnectionGene(input_node2.get_id(), output_node1.get_id(), random.random()*5, True, 4)
    connection_22 = ConnectionGene(input_node2.get_id(), output_node2.get_id(), random.random()*5, True, 5)
    connection_23 = ConnectionGene(input_node2.get_id(), output_node3.get_id(), random.random()*5, True, 6)

    initial_genome.add_connection_gene(connection_11)
    initial_genome.add_connection_gene(connection_12)
    initial_genome.add_connection_gene(connection_13)
    initial_genome.add_connection_gene(connection_21)
    initial_genome.add_connection_gene(connection_22)
    initial_genome.add_connection_gene(connection_23)

    population.append(initial_genome)
    fitness_scores.append(0)
    ff_network = FeedForwardNeuralNetwork()
    networks.append(ff_network.phenotype_from(initial_genome))

print(population)
print(fitness_scores)
print(networks)
print(brick_stats)
print(paddle_stats)
print(ball_stats)
# TODO: Add genome label
# MAIN LOOP
while running:
    # Population -> 50 genomes
    # Compute Fitness Fn.
    # Winner
    # Save Winner

    # I want to loop through the all genome's situation, only rendering best fitness, until 5 left or fitness threshold met
    # Draw background?
    # Set Generation (print)
    # Select genome if ball.alive
    # Draw Bricks[genome]
    # Draw Paddle[genome]
    # Draw Ball[genome]
    # Activate genome and get output values to see what action to take
    # If ball above screen, fitness += 1
    # If ball goes below screen, pop fitness value and genome from lists as well as bricks, paddle, ball
    # If fitness threshold met (all bricks gone) then stop generation and save the winner
    # If 5 genomes remaining, do crossover and mutation to get new population of 50 (5 + 45 = 5 + 5*9 = 5 + 4*11+1?)

    # poll for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # elif event.type == pygame.KEYDOWN:
        #     if event.key == pygame.K_RIGHT:
        #         # paddle.move(Direc.RIGHT)
        #         moving_right = True
        #     elif event.key == pygame.K_LEFT:
        #         # paddle.move(Direc.LEFT)
        #         moving_left = True
        # elif event.type == pygame.KEYUP:
        #     if event.key == pygame.K_RIGHT:
        #         moving_right = False
        #     elif event.key == pygame.K_LEFT:
        #         moving_left = False
            

    k = 0
    gen_loop = True
    while len(population) > 5:
        gen += 1
        print("Fitness scores: ", fitness_scores)
        while k < len(population):
            if len(population) == 0:
                print("NO POPUPLATION AJKDFLDJKF")
                break
            best_genome_ind = max_index(fitness_scores)
            print("Population: ", population)
            print("Pop Len: ", len(population))
            # Render game here
            screen.fill((255, 255, 255))
            screen.blit(i_bg, (0, 0))
            text = font.render("Generation: " + str(gen) + " Best Genome: " + str(best_genome_ind), True, (255, 0, 255))
            screen.blit(text, (0,0))

            if k == best_genome_ind:
                for i, brick in enumerate(brick_stats[k]):
                    for j, b in enumerate(brick):
                        if b.alive:
                            x = 33+j*b.img.get_width()
                            y = TOPLEFT_CORNER[1] - 4*BRICK_TOPLEFT_CORNER[1]+i*(b.img.get_height()-19)
                            b.x = x
                            b.y = y
                            screen.blit(b.img, (b.x, b.y))

                screen.blit(paddle_stats[k].img, (paddle_stats[k].x, paddle_stats[k].y))

            inputs = {}
            # Distance between paddle and ball multiplied by orientation of ball
            inputs[input_node1] = ball_stats[k].x_orientation*math.sqrt((paddle_stats[k].x - ball_stats[k].x)**2 + (paddle_stats[k].y - ball_stats[k].y)**2) 
            # X position of paddle
            inputs[input_node2] = paddle_stats[k].x
            network_vals = networks[k].activate(inputs, lambda x : 1 / (1 + math.exp(-x)))
            output_vals = [network_vals[3], network_vals[4], network_vals[5]]
            output_index = output_vals.index(max(output_vals))

            if output_index == 0:
                paddle_stats[k].move(Direc.LEFT)
            elif output_index == 1:
                paddle_stats[k].move(Direc.RIGHT)
            else:
                pass


            # paddle collision
            # if paddle_stats[k].collision(ball_stats[k]):
            #     print("collided with ball")


            # brick collision
            for i, brick in enumerate(brick_stats[k]):
                for j, b in enumerate (brick):
                    if b.alive and b.collision(ball_stats[k]):
                        b.alive = False
                        # score += b.score_val
                        fitness_scores[k] += b.score_val
                        if k == best_genome_ind:
                            screen.blit(ball_stats[k].img, (ball_stats[k].x, ball_stats[k].y))
                        continue

            # boundary collision
            if int(ball_stats[k].x)+ball_stats[k].VEL*math.sin(ball_stats[k].angle) <= TOPLEFT_CORNER[0]: # reflect from left
                ball_stats[k].x = TOPLEFT_CORNER[0]
                ball_stats[k].reflect("left")
            elif int(ball_stats[k].x)+ball_stats[k].VEL*math.sin(ball_stats[k].angle) >= WINDOW_WIDTH - 31 - ball_stats[k].img.get_width(): # reflect from right
                ball_stats[k].x = WINDOW_WIDTH - 31 - ball.img.get_width()
                ball_stats[k].reflect("right")
            elif int(ball_stats[k].y)+ball_stats[k].VEL*math.sin(ball_stats[k].angle) <= TOPLEFT_CORNER[1]: # reflect from top
                ball_stats[k].y = TOPLEFT_CORNER[1]
                ball_stats[k].reflect("up")
            elif int(ball_stats[k].y) >= WINDOW_HEIGHT:
                # death condition
                ball_stats[k].alive = False
                population.pop(k)
                fitness_scores.pop(k)
                networks.pop(k)
                brick_stats.pop(k)
                paddle_stats.pop(k)
                ball_stats.pop(k)
                # TODO: Does this 'continue' work?
                continue


            ball_stats[k].move()
            if k == best_genome_ind:
                screen.blit(ball_stats[k].img, (ball_stats[k].x, ball_stats[k].y))

            screen.blit(i_bg, (0, 0))

            if k == best_genome_ind:
                pygame.display.flip()

            if fitness_scores[k] >= FITNESS_SCORES_THRESHOLD:
                print("Genome ", k, " is the best genome!")
                # Save object
                with open("best_genome.txt", "w") as file:
                    file.write("Node Genes:\n")
                    for node in population[k].get_node_genes():
                        file.write(str(node)+"\n")
                    file.write("Connection Genes:\n")
                    for conn in population[k].get_connection_genes():
                        file.write(str(conn))





            fitness_scores[k] += 1
            k = (k+1) % N


            delta_time = clock.tick(60) / 1000 # limits fps to 60
            delta_time = max(0.001, min(0.1, delta_time))
            
            innovation.increment()
        # set up all lists for next generation

    new_population = population.copy()
    print("NEW POPOULATION: ", new_population, "len: ", len(new_population))
    for i in range(5):
        for j in range(5):
            # cond
            if i == j:
                continue
            for l in range(2):
                child_genome = Genome.crossover(population[i],population[j],fitness_scores[i]==fitness_scores[j])
                x = random.random()
                y = random.random()
                # Mutation rate of 0.2
                if x > 0.8:
                    if y > 0.5:
                        child_genome.add_node_mutation(innovation)
                    else:
                        child_genome.add_connection_mutation(innovation)
                population.append(child_genome)
    for i in range(5):
        if i == 4:
            child_genome = Genome.crossover(population[i],population[0],fitness_scores[i]==fitness_scores[0])
        else:
            child_genome = Genome.crossover(population[i],population[i+1],fitness_scores[i]==fitness_scores[i+1])
        x = random.random()
        y = random.random()
        # Mutation rate of 0.2
        if x > 0.8:
            if y > 0.5:
                child_genome.add_node_mutation(innovation)
            else:
                child_genome.add_connection_mutation(innovation)
        population.append(child_genome)
    print("New Generation's Population Length: ", len(population))

    brick_stats = [bricks for i in range(N)]
    paddle_stats = [Paddle() for i in range(N)]
    ball_stats = [Ball(WINDOW_WIDTH/1.5, WINDOW_HEIGHT/1.5) for i in range(N)]
    fitness_scores = [0 for i in range(N)]
    networks = []
    for i in range(N):
        network = FeedForwardNeuralNetwork()
        networks.append(network.phenotype_from(population[i]))




pygame.quit()
