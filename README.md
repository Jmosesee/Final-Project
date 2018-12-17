# Final-Project
Reinforcement Learning

This is my final project for the 2018 University of Denver Data Analytics Boot Camp.
The main prescribed requirement for the project was that it should make use of machine learning.
In class, we had covered supervised and unsupervised learning, but my background in instrumentation and controls led me to be interested in Reinforcement Learning.  Since Reinforcement Learning had not been covered in class, my presentation began with a background overview.

Although Reinforcement Learning has been famously applied to the complex environments of Atari games, I was more interested in experimenting with a simpler environment, so I focused on the CartPole game from https://gym.openai.com/envs/CartPole-v1/
This game simulates the inverted pendulum problem of classical control theory.  Using a simple test environment allowed me to closely examine the performance of the agent's algorithm.

I found ten different Q-learning solutions to the Cart-Pole environment, publicly available online.  The ten solutions have fundamental similarities outlined on slide 9 of my presentation.  I tested these agents against the CartPole-v1 environment, which requires the agent to succeed in preventing the pole from falling for 500 steps per episode, for 100 consecutive episodes.  Many of these solutions were originally intended to solve the CartPole-v0 environment, which is identical except that the game is ended after only 200 steps, rather than 500.

The summary spreadsheet lists the URL's of the 10 algorithms I found online, and their differences.  The subsequent worksheets in this file show the results of testing each of these algorithms on dedicated AWS CPU's for at least 8 hours each.  I did try testing the algorithms on GPU's but found that the GPU's were severely underutilized by the algorithms, so that the higher hardware cost failed to yield a performance improvement over CPU's.  Reinforcement learning is fundamentally iterative and sequential, which probably explains why parallel processing offered little benefit.

Six of of the ten algorithms tested failed to solve the CartPole-v1 test environment after running thousands of episodes over more than 8 hours.  One of the algorithms succeeded after 4389 episodes, but ran so slowly that the test could not be repeated.  Three of the algorithms ran fast enough that I was able to run a 30-trial test to get a probablity distribution of how many episodes were required to solve the environment.  The modes of these three probability distributions fell around 25000, 1750, and 1000 episodes.

I critically examined the fundamental characteristics that these 10 algorithms shared, and invented a new approach to using machine learning to solve the inverted pendulum problem, as exemplified by the CartPole-v1 environment.  The principles of my algorithm are outlined in the last slide of my presentation, and its code is found in FinalProject.py.  Using the same criteria against which the other 10 algorithms were tested, my algorithm typically solved the environment in less than 70 episodes.

I am pleased that I was able to come up with an algorithm that performed more than 10 times better than any of the existing publicly available algorithms I could find.






