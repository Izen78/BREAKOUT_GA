# TODO: Deltatime
# TODO: Render all bricks

import pygame
import os
from typing import Type

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
    img = i_rb
    
    def __init__(self) -> None:
        super().__init__(self.score_val)

class Brown_Brick(Brick):
    score_val = 50

class Yellow_Brick(Brick):
    score_val = 25

class Green_Brick(Brick):
    score_val = 10

class Blue_Brick(Brick):
    score_val = 5





class Ball:
    pass

class Bar:
    pass

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()
running = True

# Initial Screen
screen.blit(i_bg, (0, 0))

# 18 red bricks across (12x32) first brick starts at x=32, y=64
rbs = []
for i in range(18):
    rbs.append(Red_Brick())
for i, rb in enumerate(rbs):
    screen.blit(rb.img, ((33+i*rb.img.get_width(),TOPLEFT_CORNER[1]-4*BRICK_TOPLEFT_CORNER[1])))

while running:
    # poll for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Render game here

    pygame.display.flip()

    clock.tick(60) # limits fps to 60

pygame.quit()
