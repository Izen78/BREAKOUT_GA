# TODO:
# Ball Physics 
# Ball collision with grey (DOESN'T WORK SINCE WE'RE GOING EVERY 5 PIXELS AND DON'T CAPTURE THE CRITICAL COLLISION POINTS)
# Collision with Bricks which removes Bricks from being rendered

import math
import pygame
import os
from typing import Type
import random

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 400
TOPLEFT_CORNER = (33, 60)
BRICK_TOPLEFT_CORNER= (0, -10)

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

class Brick:
    score_val = None
    alive = True
    img = None

    def __init__(self, score_val) -> None:
        self.score_val = score_val

    def collision(self, ball: type) -> bool:
        pass

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

class Ball:
    angle = 0 # radians, up is [0, pi] and down is (pi, 2pi)
    VEL = 5 
    img = i_ball

    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.angle = random.uniform(0, 2*math.pi)

    def move(self) -> None:
        self.x += self.VEL * math.cos(self.angle)
        self.y -= self.VEL * math.sin(self.angle)

    def reflect(self, dir) -> None:
        if dir == "left":
            print("left")
            orientation = -1 if self.angle <= math.pi else 1
            self.angle += orientation*(math.pi/2)
        elif dir == "right":
            print("right")
            orientation = 1 if self.angle <= math.pi else -1
            self.angle += orientation*(math.pi/2)
        elif dir == "up":
            print("up")
            orientation = -1 if self.angle <= math.pi/2 else 1
            self.angle += orientation*(math.pi/2)




        

class Bar:
    pass

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()
running = True

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
for i, brick in enumerate(bricks):
    for j, b in enumerate(brick):
        screen.blit(b.img, ((33+j*b.img.get_width(), TOPLEFT_CORNER[1]-4*BRICK_TOPLEFT_CORNER[1]+i*(b.img.get_height()-19))))

ball = Ball(WINDOW_WIDTH/2, WINDOW_HEIGHT/2)

delta_time = 0.1
while running:

    # poll for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Render game here
    screen.fill((255, 255, 255))
    screen.blit(i_bg, (0, 0))
    bricks = [rbs, obs, br_bs, ybs, gbs, bbs]
    for i, brick in enumerate(bricks):
        for j, b in enumerate(brick):
            screen.blit(b.img, ((33+j*b.img.get_width(), TOPLEFT_CORNER[1]-4*BRICK_TOPLEFT_CORNER[1]+i*(b.img.get_height()-19))))
    screen.blit(ball.img, (ball.x, ball.y))

    # brick collision

    # boundary collision
    if int(ball.x) == TOPLEFT_CORNER[0]: # reflect from left
        print("1")
        ball.reflect("left")
    elif int(ball.x) == WINDOW_WIDTH - 31 - ball.img.get_width(): # reflect from right
        print("2")
        ball.reflect("right")
    elif int(ball.y) == TOPLEFT_CORNER[1]: # reflect from top
        print("3")
        ball.reflect("up")

    print(int(ball.x)) # BUG: LEFT OFF HERE
    ball.move()

    screen.blit(i_bg, (0, 0))

    pygame.display.flip()

    delta_time = clock.tick(60) / 1000 # limits fps to 60
    delta_time = max(0.001, min(0.1, delta_time))

pygame.quit()
