import pygame
import os

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 400
# Import Images

i_bg = pygame.image.load("imgs/bg.png")
i_one = pygame.image.load("imgs/one.png")
class Brick:
    pass

class Ball:
    pass

class Bar:
    pass

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()
running = True

screen.blit(i_bg, (0, 0))

while running:
    # poll for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Render game here

    pygame.display.flip()

    clock.tick(60) # limits fps to 60

pygame.quit()
