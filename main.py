# TODO:
# Angle threshold
# Paddle Collision 
# Separate Ball Inputs x, y, vel_x, vel_y

import math
import pygame
import os
from typing import TYPE_CHECKING,  Type
import random
from enum import Enum

from neat import *

import copy
import pickle

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
        self.angle =  random.uniform(3*math.pi-math.pi/4, 3*math.pi+math.pi/4)

    def move(self) -> None:
        self.x += self.VEL * math.cos(self.angle)
        self.y -= self.VEL * math.sin(self.angle)

    # def reflect(self, direc) -> None:
    #     if self.y_orientation == 1:
    #         if self.x_orientation == 1:
    #             if self.angle <= self.ANGLE_THRESHOLD:
    #                 self.angle += self.ANGLE_THRESHOLD
    #             elif self.angle <= math.pi/2 and self.angle >= math.pi/2-self.ANGLE_THRESHOLD:
    #                 self.angle -= self.ANGLE_THRESHOLD
    #         elif self.x_orientation == -1:
    #             if self.angle <= math.pi and self.angle >= math.pi-self.ANGLE_THRESHOLD:
    #                 self.angle -= self.ANGLE_THRESHOLD
    #             elif self.angle <= math.pi/2+self.ANGLE_THRESHOLD and self.angle >= math.pi/2:
    #                 self.angle += self.ANGLE_THRESHOLD
    #     elif self.y_orientation == -1:
    #         if self.x_orientation == 1:
    #             if self.angle <= 2*math.pi and self.angle >= 2*math.pi-self.ANGLE_THRESHOLD:
    #                 self.angle -= self.ANGLE_THRESHOLD
    #             elif self.angle <= 3*math.pi/2+self.ANGLE_THRESHOLD and self.angle >= 3*math.pi/2:
    #                 self.angle += self.ANGLE_THRESHOLD
    #         elif self.x_orientation == -1:
    #             if self.angle <= math.pi+self.ANGLE_THRESHOLD and self.angle >= math.pi:
    #                 self.angle += self.ANGLE_THRESHOLD
    #             elif self.angle <= 3*math.pi/2 and self.angle >= 3*math.pi/2-self.ANGLE_THRESHOLD:
    #                 self.angle -= self.ANGLE_THRESHOLD
    #
    #     if direc == "left":
    #         if self.angle <= math.pi and self.angle >= math.pi/2:
    #             x_ang = math.pi - self.angle
    #             self.angle = x_ang
    #             self.x_orientation = 1
    #             self.y_orientation = 1
    #         elif self.angle >= math.pi and self.angle <= 3*math.pi/2:
    #             x_ang = self.angle - math.pi
    #             self.angle = 2*math.pi - x_ang
    #             self.x_orientation = 1
    #             self.y_orientation = -1
    #     elif direc == "right":
    #         if self.angle <= math.pi/2 and self.angle >= 0:
    #             x_ang = self.angle
    #             self.angle = math.pi - x_ang
    #             self.x_orientation = -1
    #             self.y_orientation = 1
    #         elif self.angle <= 2*math.pi and self.angle >= 3*math.pi/2:
    #             x_ang = 2*math.pi - self.angle
    #             self.angle = math.pi + x_ang
    #             self.y_orientation = -1
    #             self.x_orientation = -1
    #     elif direc == "up":
    #         if self.angle >= 0 and self.angle <= math.pi/2:
    #             x_ang = self.angle
    #             self.angle = 2*math.pi - x_ang
    #             self.x_orientation = 1
    #             self.y_orientation = -1
    #         elif self.angle <= math.pi and self.angle >= math.pi/2:
    #             x_ang = math.pi - self.angle
    #             self.angle = math.pi + x_ang
    #             self.x_orientation = -1
    #             self.y_orientation = -1
    #     elif direc == "down":
    #         if self.angle >= 3*math.pi/2 and self.angle <= 2*math.pi:
    #             x_ang = 2*math.pi - self.angle
    #             self.angle = x_ang
    #             self.x_orientation = 1
    #             self.y_orientation = 1
    #         if self.angle >= math.pi and self.angle <= 3*math.pi/2:
    #             x_ang = self.angle - math.pi
    #             self.angle = math.pi - x_ang
    #             self.x_orientation = -1
    #             self.y_orientation = 1
    #
    #
    #     self.angle = self.angle % (2*math.pi)
    #     self.move()
    def avoid_flat_angles(self):
        if abs(math.cos(self.angle)) < 0.05:  # near vertical
            self.angle += self.ANGLE_THRESHOLD
        elif abs(math.sin(self.angle)) < 0.05:  # near horizontal
            self.angle += self.ANGLE_THRESHOLD
        self.angle %= 2 * math.pi

    def enforce_angle_threshold(self):
        """Avoids angles too close to purely horizontal or vertical."""
        # Angle is in radians: [0, 2π)
        flat_horizontal_angles = [0, math.pi, 2 * math.pi]
        flat_vertical_angles = [math.pi / 2, 3 * math.pi / 2]

        for a in flat_horizontal_angles:
            if abs(self.angle - a) < self.ANGLE_THRESHOLD:
                self.angle += 2*self.ANGLE_THRESHOLD
                self.reflect("up")
                break

        for a in flat_vertical_angles:
            if abs(self.angle - a) < self.ANGLE_THRESHOLD:
                self.angle += self.ANGLE_THRESHOLD
                break

        self.angle %= 2 * math.pi

    def reflect(self, direc) -> None:
        if direc == "left" or direc == "right":
            self.angle = math.pi - self.angle
            self.x_orientation *= -1
            self.enforce_angle_threshold()
        elif direc == "down":
            self.angle = -self.angle
            self.y_orientation *= -1
            self.enforce_angle_threshold()
        elif direc == "up":
            self.angle = -self.angle
            self.y_orientation = 1
            self.enforce_angle_threshold()

        self.angle %= 2 * math.pi
        self.move()

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

    # def collision(self, ball: Type[Ball]) -> bool:
    #     next_bx = int(ball.x) + ball.VEL*math.cos(ball.angle)
    #     next_by = int(ball.y) + ball.VEL*math.sin(ball.angle)
    #
    #     if next_bx <= self.x + self.img.get_width() and next_bx >= self.x:
    #         # Either directly above or below brick
    #         if next_by <= self.y + self.img.get_height() and next_by >= self.y and ball.y_orientation == 1: # reflect from top
    #             # BELOW
    #             # ball.y = (self.y + self.img.get_height())*math.sin(ball.angle)
    #             ball.reflect("up")
    #             print("ball reflecting from up")
    #             return True
    #         elif next_by+ball.img.get_height() >= self.y and next_by+ball.img.get_height() <= self.y + self.img.get_height() and ball.y_orientation == -1:
    #             # ABOVE
    #             ball.reflect("down")
    #             print("ball reflecting from down")
    #             return True
    #     # elif next_by >= self.y and next_by <= self.y + self.img.get_height():
    #     #     # Either directly left or right of brick
    #     #     if next_bx <= self.x + self.img.get_width() and next_bx >= self.x:
    #     #         # REFLECT FROM THE LEFT
    #     #         ball.reflect("left")
    #     #         return True
    #     #     elif next_bx+ball.img.get_width() >= self.x and next_bx+ball.img.get_width() <= self.x + self.img.get_width():
    #     #         # REFLECT FROM RIGHT
    #     #         ball.reflect("right")
    #     #         return True
    #     return False

    def collision(self, ball: Type[Ball]) -> bool:
        future_x = ball.x + ball.VEL * math.cos(ball.angle)
        future_y = ball.y - ball.VEL * math.sin(ball.angle)  # remember y is flipped in Pygame

        # Get ball bounding box
        ball_rect = pygame.Rect(future_x, future_y, ball.img.get_width(), ball.img.get_height())
        paddle_rect = pygame.Rect(self.x, self.y, self.img.get_width(), self.img.get_height())

        if ball_rect.colliderect(paddle_rect) and ball.y_orientation == 1:
            ball.reflect("up")
            # Push the ball back to just above the paddle to avoid double-collision next frame
            ball.y = self.y - ball.img.get_height() - 1
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

# ball = Ball(WINDOW_WIDTH/1.5, WINDOW_HEIGHT/1.5)
# ball = Ball(WINDOW_WIDTH/1.5, 70)
# paddle = Paddle()
delta_time = 0.1
moving_right = False
moving_left = False
score = 0
FITNESS_SCORES_THRESHOLD = 4770 + 60*5 # all blocks + 5 seconds

gen = 1
N = 50
population = []
fitness_scores = []
networks = []
brick_stats = [bricks for i in range(N)]
paddle_stats = [Paddle() for i in range(N)]
ball_stats = [Ball(WINDOW_WIDTH/1.5, WINDOW_HEIGHT/1.5) for i in range(N)]

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
for i in range(N):
    initial_genome = Genome()

    initial_genome.add_node_gene(input_node1)
    initial_genome.add_node_gene(input_node2)
    initial_genome.add_node_gene(input_node3)
    initial_genome.add_node_gene(input_node4)
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
    fitness_scores.append(0)
    ff_network = FeedForwardNeuralNetwork()
    networks.append(ff_network.phenotype_from(initial_genome))

# print(population)
# print(fitness_scores)
# print(networks)
# print(brick_stats)
# print(paddle_stats)
# print(ball_stats)

# TODO: Add genome label
# MAIN LOOP

# while running:
#     # I want to loop through the all genome's situation, only rendering best fitness, until 5 left or fitness threshold met
#     # Draw background?
#     # Set Generation (print)
#     # Select genome if ball.alive
#     # Draw Bricks[genome]
#     # Draw Paddle[genome]
#     # Draw Ball[genome]
#     # Activate genome and get output values to see what action to take
#     # If ball above screen, fitness += 1
#     # If ball goes below screen, pop fitness value and genome from lists as well as bricks, paddle, ball
#     # If fitness threshold met (all bricks gone) then stop generation and save the winner
#     # If 5 genomes remaining, do crossover and mutation to get new population of 50 (5 + 45 = 5 + 5*9 = 5 + 4*11+1?)
#
#     # poll for events
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#             
#     k = 0
#     gen_loop = True
#     while len(population) > 5:
#         print("Fitness scores: ", fitness_scores)
#         while k < len(population):
#             if len(population) == 0:
#                 print("NO POPUPLATION AJKDFLDJKF")
#                 break
#             best_genome_ind = max_index(fitness_scores)
#             print("Population: ", population)
#             print("Pop Len: ", len(population))
#             # Render game here
#             screen.fill((255, 255, 255))
#             screen.blit(i_bg, (0, 0))
#             text = font.render("Generation: " + str(gen) + " Best Genome: " + str(best_genome_ind), True, (255, 0, 255))
#             screen.blit(text, (0,0))
#
#             inputs = {
#                 input_node1: ball_stats[k].x_orientation * math.sqrt(
#                     (paddle_stats[k].x - ball_stats[k].x)**2
#                 +   (paddle_stats[k].y - ball_stats[k].y)**2),
#                 input_node2: paddle_stats[k].x
#             }
#
#             try:
#                 output_vals = networks[k].activate(inputs, lambda x : 1 / (1+math.exp(-x)))
#                 outputs = [output_vals[3], output_vals[4], output_vals[5]]
#             except KeyError:
#                 k = (k+1) % len(population)
#                 continue
#             action = outputs.index(max(outputs))
#             if action == 0:
#                 paddle_stats[k].move(Direc.LEFT)
#             elif action == 1:
#                 paddle_stats[k].move(Direc.RIGHT)
#             # if k == best_genome_ind:
#
#             for row in brick_stats[k]:
#                 for b in row:
#                     if b.alive and b.collision(ball_stats[k]):
#                         b.alive = False
#                         fitness_scores[k] += b.score_val
#
#             ball = ball_stats[k]
#             if int(ball.y) >= WINDOW_HEIGHT:
#                 population.pop(k)
#                 fitness_scores.pop(k)
#                 networks.pop(k)
#                 brick_stats.pop(k)
#                 paddle_stats.pop(k)
#                 ball_stats.pop(k)
#                 continue
#             
#             ball.move()
#             fitness_scores[k] += 1
#             k = (k+1)%len(population)
#
#             if k == best_genome_ind:
#                 for row in brick_stats[k]:
#                     for b in row:
#                         if b.alive:
#                             screen.blit(b.img, (b.x, b.y))
#                 screen.blit(paddle_stats[k].img, (paddle_stats[k].x,paddle_stats[k].y))
#                 screen.blit(ball_stats[k].img, (ball_stats[k].x,ball_stats[k].y))
#                 pygame.display.flip()
#
#             clock.tick(60)
#             innovation.increment()
            #     for i, brick in enumerate(brick_stats[k]):
            #         for j, b in enumerate(brick):
            #             if b.alive:
            #                 x = 33+j*b.img.get_width()
            #                 y = TOPLEFT_CORNER[1] - 4*BRICK_TOPLEFT_CORNER[1]+i*(b.img.get_height()-19)
            #                 b.x = x
            #                 b.y = y
            #                 screen.blit(b.img, (b.x, b.y))
            #
            #     screen.blit(paddle_stats[k].img, (paddle_stats[k].x, paddle_stats[k].y))
            #
            # inputs = {}
            # # Distance between paddle and ball multiplied by orientation of ball
            # inputs[input_node1] = ball_stats[k].x_orientation*math.sqrt((paddle_stats[k].x - ball_stats[k].x)**2 + (paddle_stats[k].y - ball_stats[k].y)**2) 
            # # X position of paddle
            # inputs[input_node2] = paddle_stats[k].x
            # network_vals = networks[k].activate(inputs, lambda x : 1 / (1 + math.exp(-x)))
            # output_vals = [network_vals[3], network_vals[4], network_vals[5]]
            # output_index = output_vals.index(max(output_vals))
            #
            # if output_index == 0:
            #     paddle_stats[k].move(Direc.LEFT)
            # elif output_index == 1:
            #     paddle_stats[k].move(Direc.RIGHT)
            # else:
            #     pass
            #

            # paddle collision
            # if paddle_stats[k].collision(ball_stats[k]):
            #     print("collided with ball")


            # brick collision
            # for i, brick in enumerate(brick_stats[k]):
            #     for j, b in enumerate (brick):
            #         if b.alive and b.collision(ball_stats[k]):
            #             b.alive = False
            #             # score += b.score_val
            #             fitness_scores[k] += b.score_val
            #             if k == best_genome_ind:
            #                 screen.blit(ball_stats[k].img, (ball_stats[k].x, ball_stats[k].y))
            #             continue

            # boundary collision
            # if int(ball_stats[k].x)+ball_stats[k].VEL*math.sin(ball_stats[k].angle) <= TOPLEFT_CORNER[0]: # reflect from left
            #     ball_stats[k].x = TOPLEFT_CORNER[0]
            #     ball_stats[k].reflect("left")
            # elif int(ball_stats[k].x)+ball_stats[k].VEL*math.sin(ball_stats[k].angle) >= WINDOW_WIDTH - 31 - ball_stats[k].img.get_width(): # reflect from right
            #     ball_stats[k].x = WINDOW_WIDTH - 31 - ball.img.get_width()
            #     ball_stats[k].reflect("right")
            # elif int(ball_stats[k].y)+ball_stats[k].VEL*math.sin(ball_stats[k].angle) <= TOPLEFT_CORNER[1]: # reflect from top
            #     ball_stats[k].y = TOPLEFT_CORNER[1]
            #     ball_stats[k].reflect("up")
            # elif int(ball_stats[k].y) >= WINDOW_HEIGHT:
            #     # death condition
            #     ball_stats[k].alive = False
            #     population.pop(k)
            #     fitness_scores.pop(k)
            #     networks.pop(k)
            #     brick_stats.pop(k)
            #     paddle_stats.pop(k)
            #     ball_stats.pop(k)
            #     # TODO: Does this 'continue' work?
            #     continue
            #
            #
            # ball_stats[k].move()
            # if k == best_genome_ind:
            #     screen.blit(ball_stats[k].img, (ball_stats[k].x, ball_stats[k].y))
            #
            # screen.blit(i_bg, (0, 0))
            #
            # if k == best_genome_ind:
            #     pygame.display.flip()
            #
            # if fitness_scores[k] >= FITNESS_SCORES_THRESHOLD:
            #     print("Genome ", k, " is the best genome!")
            #     # Save object
            #     with open("best_genome.txt", "w") as file:
            #         file.write("Node Genes:\n")
            #         for node in population[k].get_node_genes():
            #             file.write(str(node)+"\n")
            #         file.write("Connection Genes:\n")
            #         for conn in population[k].get_connection_genes():
            #             file.write(str(conn))
            #
            #
            #
            #
            #
            # fitness_scores[k] += 1
            # k = (k+1) % N
            #

            # delta_time = clock.tick(60) / 1000 # limits fps to 60
            # delta_time = max(0.001, min(0.1, delta_time))
            
            # innovation.increment()
        # set up all lists for next generation

    # new_population = population.copy()
    # print("NEW POPOULATION: ", new_population, "len: ", len(new_population))
    # for i in range(5):
    #     for j in range(5):
    #         # cond
    #         if i == j:
    #             continue
    #         for l in range(2):
    #             child_genome = Genome.crossover(population[i],population[j],fitness_scores[i]==fitness_scores[j])
    #             x = random.random()
    #             y = random.random()
    #             # Mutation rate of 0.2
    #             if x > 0.8:
    #                 if y > 0.5:
    #                     child_genome.add_node_mutation(innovation)
    #                 else:
    #                     child_genome.add_connection_mutation(innovation)
    #             population.append(child_genome)
    # for i in range(5):
    #     if i == 4:
    #         child_genome = Genome.crossover(population[i],population[0],fitness_scores[i]==fitness_scores[0])
    #     else:
    #         child_genome = Genome.crossover(population[i],population[i+1],fitness_scores[i]==fitness_scores[i+1])
    #     x = random.random()
    #     y = random.random()
    #     # Mutation rate of 0.2
    #     if x > 0.8:
    #         if y > 0.5:
    #             child_genome.add_node_mutation(innovation)
    #         else:
    #             child_genome.add_connection_mutation(innovation)
    #     population.append(child_genome)
    # print("New Generation's Population Length: ", len(population))
    #
    # brick_stats = [bricks for i in range(N)]
    # paddle_stats = [Paddle() for i in range(N)]
    # ball_stats = [Ball(WINDOW_WIDTH/1.5, WINDOW_HEIGHT/1.5) for i in range(N)]
    # fitness_scores = [0 for i in range(N)]
    # networks = []
    # for i in range(N):
    #     network = FeedForwardNeuralNetwork()
    #     networks.append(network.phenotype_from(population[i]))

def enforce_output_connectivity(genome: Genome, input_ids: set[int], output_ids: set[int], innovation: InnovationTracker):
    connected_outputs = {conn.get_out_node() for conn in genome.get_connection_genes() if conn.is_expressed()}
    for output_id in output_ids:
        if output_id not in connected_outputs:
            # Randomly pick an input node to connect from
            input_id = random.choice(list(input_ids))
            conn = ConnectionGene(input_id, output_id, random.uniform(-1, 1), True, innovation.get_innovation())
            genome.add_connection_gene(conn)
            innovation.increment()
def render(agent):
    screen.fill((255,255,255))
    i_bg.set_colorkey((0,0,0))
    # Draw Background
    screen.blit(i_bg, (0, 0))

    # Draw Text

    text = font.render("Generation: " + str(gen) + " Best Genome: " + str(agent["id"]), True, (255, 0, 255))
    screen.blit(text, (0,0))

    # Draw Bricks

    # for row in agent["bricks"]:
    #     for b in row:
    #         if b.alive:
    #             screen.blit(b.img, (b.x, b.y))

    for i, brick in enumerate(agent["bricks"]):
        for j, b in enumerate(brick):
            if b.alive:
                x = 33+j*b.img.get_width()
                y = TOPLEFT_CORNER[1] - 4*BRICK_TOPLEFT_CORNER[1]+i*(b.img.get_height()-19)
                b.x = x
                b.y = y
                screen.blit(b.img, (b.x, b.y))
 
    # Draw Ball
    screen.blit(agent["ball"].img, (agent["ball"].x,agent["ball"].y))
    # print("ANGLE: ", agent["ball"].angle)

    # Draw Paddle
    screen.blit(agent["paddle"].img, (agent["paddle"].x,agent["paddle"].y))

    pygame.display.flip()
    delta_time = clock.tick(60) / 1000 # limits fps to 60
    delta_time = max(0.001, min(0.1, delta_time))

def eval_genomes(genomes):
    agents = []

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
    id = 0
    for genome in genomes:
        net = FeedForwardNeuralNetwork()
        net = net.phenotype_from(genome)
        paddle = Paddle()
        ball = Ball(WINDOW_WIDTH/1.5, WINDOW_HEIGHT/1.5)
        
        agents.append({
            "genome":genome,
            "net": net,
            "paddle": paddle,
            "ball": ball,
            "bricks": bricks,
            "fitness": 0.0,
            "alive": True,
            "id": id
        })
        id += 1

    while any(agent["alive"] for agent in agents):
        for agent in agents:
            if not agent["alive"]:
                continue

            inputs = {input_node1: agent["ball"].x_orientation*math.sqrt((agent["paddle"].x - agent["ball"].x)**2 \
                                            + (agent["paddle"].y - agent["ball"].y)**2),
                      input_node2: agent["paddle"].x}
            values = agent["net"].activate(inputs, lambda x : 1 / (1 + math.exp(-x)))
            output_ids = []
            for node in agent["net"].outputs:
                output_ids.append(node.get_id())

            outputs = {k: values[k] for k in output_ids if k in values}
            if not outputs:
                print(f"[WARNING] No outputs found for genome {agent['id']}")
            # outputs[3] = values[3]
            # outputs[4] = values[4]
            # outputs[5] = values[5]
            best_output = max(outputs, key=outputs.get)
            if best_output == 5:
                agent["paddle"].move(Direc.LEFT)
            elif best_output == 6:
                agent["paddle"].move(Direc.RIGHT)

            agent["ball"].move()

            # Paddle Collision
            if agent["paddle"].collision(agent["ball"]):
                print(f"paddle {agent["id"]} has collided with ball")

            # Brick Collision

            for i, brick in enumerate(agent["bricks"]):
                for j, b in enumerate (brick):
                    if b.alive and b.collision(agent["ball"]):
                        b.alive = False
                        # score += b.score_val
                        agent["fitness"] += b.score_val
                        print(f"agent {agent["id"]} hit brick ({i}, {j})")

            # Boundary Collision
            if int(agent["ball"].x)+agent["ball"].VEL*math.sin(agent["ball"].angle) <= TOPLEFT_CORNER[0]: # reflect from left
                agent["ball"].x = TOPLEFT_CORNER[0]
                agent["ball"].reflect("left")
            elif int(agent["ball"].x)+agent["ball"].VEL*math.sin(agent["ball"].angle) >= WINDOW_WIDTH - 31 - agent["ball"].img.get_width(): # reflect from right
                agent["ball"].x = WINDOW_WIDTH - 31 - agent["ball"].img.get_width()
                agent["ball"].reflect("right")
            elif int(agent["ball"].y)+agent["ball"].VEL*math.sin(agent["ball"].angle) <= TOPLEFT_CORNER[1]: # reflect from top
                agent["ball"].y = TOPLEFT_CORNER[1]
                agent["ball"].reflect("up")
            

            if agent["ball"].y >= WINDOW_HEIGHT:
                agent["alive"] = False
            else:
                agent["fitness"] += 1
        # w = 0
        # for agent in agents:
        #     if not agent["alive"]:
        #         w += 1
        # print(w)
        best_agent = max(agents, key = lambda a: a["fitness"])
        render(best_agent)


    return agents
        
if __name__ == "__main__":
    MAX_GENERATIONS = 75
    ELITE_N = 5
    while gen < MAX_GENERATIONS:
        eval_population = eval_genomes(population)
        # all genomes 'dead' now

        sorted_population = sorted(eval_population, key=lambda g: g["fitness"], reverse=True)
        elites = sorted_population[:ELITE_N]

        print(f"Generation {gen} | Best fitness: {elites[0]["fitness"]:.2f}")
        elites[0]["genome"].print_genome()

        if elites[0]["fitness"] > FITNESS_SCORES_THRESHOLD:
            with open("best_genome.pkl","wb") as file:
                pickle.dump(elites[0]["genome"], file)

        new_population = [elite.copy() for elite in elites]
        new_population = [elite["genome"] for elite in elites]

        # Pick the top 5 genomes
        # Randomly cross over and mutate them until length of new population is N

        while len(new_population) < N:
            # Parent selection: random from top K
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)

            # Choose fitter parent
            if parent1["fitness"] >= parent2["fitness"]:
                fitter, weaker = parent1, parent2
            else:
                fitter, weaker = parent2, parent1

            equal_fitness = abs(parent1["fitness"] - parent2["fitness"]) < 1e-3

            # Crossover to create child
            child = Genome.crossover(fitter["genome"].clone(), weaker["genome"].clone(), equal_fitness)
            enforce_output_connectivity(child, input_ids={1,2,3,4}, output_ids={5,6,7}, innovation=innovation)

            # Mutate child (add connection or node)
            if random.random() < 0.8:
                child.add_connection_mutation(innovation)
            if random.random() < 0.8:
                child.add_node_mutation(innovation)

            # child.print_genome()
            new_population.append(child)

        population = new_population
        gen += 1

    pygame.quit()
