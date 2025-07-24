# BREAKOUT GENETIC ALGORITHM

NeuroEvolution of Augmenting Topologies (NEAT) implementation of the Atari Breakout Game. Using libraries pygame and pytorch.

## Functionality:

This project does not include *speciation* from the original paper and the collision of the ball can be quite buggy.

**The collision of the bricks is global for each genomes**. i.e. The bricks will disappear if any genome has hit it and will be invisible if the brick has not hit it.

The best performing neural network is displayed after each generation which shows the *crossover* and *mutation* functions which uses the same logic and parameters as the paper. 

## VIDEO:


## References
Original NEAT Paper: https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
