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
        self.angle =  random.uniform(0, math.pi)

    def move(self) -> None:
        self.x += self.VEL * math.cos(self.angle)
        self.y -= self.VEL * math.sin(self.angle)

    def reflect(self, direc) -> None:
        if direc == "left":
            print("left")
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
            print("right")
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
            print("up")
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
            print("down")
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
            paddle.x = max(paddle.x - self.VEL, TOPLEFT_CORNER[0])
        elif direc == Direc.RIGHT:
            paddle.x = min(paddle.x + self.VEL, WINDOW_WIDTH - TOPLEFT_CORNER[0] - self.img.get_width() + 2)

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

ball = Ball(WINDOW_WIDTH/1.5, WINDOW_HEIGHT/1.5)
# ball = Ball(WINDOW_WIDTH/1.5, 70)
paddle = Paddle()

delta_time = 0.1
moving_right = False
moving_left = False
score = 0

# MAIN LOOP
while running:

    # poll for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                # paddle.move(Direc.RIGHT)
                moving_right = True
            elif event.key == pygame.K_LEFT:
                # paddle.move(Direc.LEFT)
                moving_left = True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_RIGHT:
                moving_right = False
            elif event.key == pygame.K_LEFT:
                moving_left = False
            

    # Render game here
    screen.fill((255, 255, 255))
    screen.blit(i_bg, (0, 0))
    for i, brick in enumerate(bricks):
        for j, b in enumerate(brick):
            if b.alive:
                x = 33+j*b.img.get_width()
                y = TOPLEFT_CORNER[1] - 4*BRICK_TOPLEFT_CORNER[1]+i*(b.img.get_height()-19)
                b.x = x
                b.y = y
                screen.blit(b.img, (b.x, b.y))
    screen.blit(ball.img, (ball.x, ball.y))

    if moving_right:
        paddle.move(Direc.RIGHT)
    elif moving_left:
        paddle.move(Direc.LEFT)

    screen.blit(paddle.img, (paddle.x, paddle.y))

    # paddle collision
    if paddle.collision(ball):
        print("collided with ball")


    # brick collision
    for i, brick in enumerate(bricks):
        for j, b in enumerate (brick):
            if b.alive and b.collision(ball):
                b.alive = False
                score += b.score_val
                print("score: ", score)
                screen.blit(ball.img, (ball.x, ball.y))
                continue

    # boundary collision
    if int(ball.x)+ball.VEL*math.sin(ball.angle) <= TOPLEFT_CORNER[0]: # reflect from left
        ball.x = TOPLEFT_CORNER[0]
        ball.reflect("left")
    elif int(ball.x)+ball.VEL*math.sin(ball.angle) >= WINDOW_WIDTH - 31 - ball.img.get_width(): # reflect from right
        ball.x = WINDOW_WIDTH - 31 - ball.img.get_width()
        ball.reflect("right")
    elif int(ball.y)+ball.VEL*math.sin(ball.angle) <= TOPLEFT_CORNER[1]: # reflect from top
        ball.y = TOPLEFT_CORNER[1]
        ball.reflect("up")
    elif int(ball.y) >= WINDOW_HEIGHT:
        # death condition
        ball.alive = False

    ball.move()

    screen.blit(i_bg, (0, 0))

    pygame.display.flip()

    delta_time = clock.tick(60) / 1000 # limits fps to 60
    delta_time = max(0.001, min(0.1, delta_time))

pygame.quit()
