# This is Josh Moses' Final Project for the 2018 University of Denver Data Analytics Boot Camp
# The topic of the project is Reinforcement Learning
# The CartPole-v1 environment from https://gym.openai.com/envs/CartPole-v1/ is used to test a new Reinforcement Learning algorithm



import gym
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import csv
import time

env = gym.make('CartPole-v1')
learning_rate = 0.0015
gamma = 0.98        # Todo: Try 0.99
n_win_score=495
episodes = 600

# These hyperparameters are typical in Q-learning but not needed for my new algorithm
# exploration_rate = 1.0
# exploration_min = 0.01
# exploration_decay = 0.995
# sample_batch_size = 32
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Todo: Try the model from D
def build_model():
    # Build a neural network
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation ='relu'))  #Todo: Try 16, 12
    model.add(Dense(48, activation ='relu'))                        #Todo: Try 16, 12
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    return model


def decide_action(state, model):
# The new algorithm is simpler.
# The standard approach first makes a choice between "exploit vs explore"
# If "explore" is selected, the agent will act randomly
# But this standard approach will not yield good performance for the CartPole game
# The new algorithm eliminates the need for the, "exploit vs explore" tradeoff
# It simply chooses the action with the highest expected reward
# Since all rewards are negative, the action with the highest expected reward will always be the least well known action.
# Thus the incentive to "explore" is already inherently built into the reward system.
    # if np.random.rand() <= exploration_rate:
    #     # The agent acts randomly
    #     return env.action_space.sample()
    # Predict the reward value based on the given state
    act_values = model.predict(state)
    # Pick the action based on the predicted reward
    return np.argmax(act_values[0])

# Run one episode of the game
def run_episode(model):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    index = 0
    # episode_memory remembers the steps and results of this one episode only
    episode_memory = deque(maxlen=2000)

    while not done:
        # To animate the game, uncomment the following line.  (Execution will be much slower.)
        # env.render()
        action = decide_action(state, model)
        next_state, reward, done, _=env.step(action)  # Execute action
        next_state = np.reshape(next_state, [1, state_size])
        episode_memory.append((state, action, reward, next_state, done))
        state = next_state
        index += 1
        if (reward != 1):   # This never happens
            print(f"Episode {e}# Score:{reward}")
    return (index, episode_memory)

def train_agent(episode_memory, model):
# Use the memory of one episode of the game to train the agent based the events of that episode
# Assign a reward of -1 to the final step of the game
    reward = -1
    while True:
        # Iterate through the entire memory of the game, starting from the final step
        try:
            (state, action, _, next_state, _) = episode_memory.pop()
        except:
            break
        target = reward
        # Get the current prediction for this state, only so that we don't change it for the untested action
        target_f = model.predict(state)
        target_f[0][action] = target
        # Train the model to expect this reware, for this (state, action)
        model.fit(state, target_f, epochs=1, verbose=0)
        # Apply a discount factor (gamma) to the penalty assigned for the previous action
        reward *= gamma

# Test the agent for episodes
def test_agent(episodes):
    total_score = 0
    model = build_model()
    # model.reset_states()
    scores = []
    # Keep track of the last 100 scores.
    # The agent succeeds in solving the game by achieving an average score of at least n_win_score over the last 100 episodes
    last_scores = deque(maxlen=100)
    for e in range(episodes):
        (score, episode_memory) = run_episode(model)
        total_score += score
        scores.append(score)
        last_scores.append(score)
        print(f"Episode {e}# Score:{score}")
        with open('log.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow((e, score))
        # Only a score of 500 is a win.  Anything less is a failure.
        # Train to avoid the mistakes that led to failure.  If the episode was won, no retraining is needed.
        if score<500:
            train_agent(episode_memory, model)

        # Report result
        mean_score = np.mean(last_scores)
        if mean_score >= n_win_score and e >= 100:
            print('Ran {} episodes. Solved after {} episodes ✔ Total score: {}'.format(e, e - 100, total_score))
            break;
        # The traditional algorithm will, at this point, "replay," a random selection of episodes from memory, to improve the model's prediction of those results.
        # My new algorithm does not need this replay step.
        # replay(sample_batch_size)
    return (e-100, total_score)

trials = 30
results = []
overall_total_score = 0
start_time = time.time()
for trial in range(trials):
    print (f'Trial: {trial}')
    (score, total_score) = test_agent(episodes)
    results.append(score)
    overall_total_score += total_score

print (f'Ticks per second: {overall_total_score / (time.time() - start_time)}')
# print (f'Overall total score: {overall_total_score}')
print ('Results: ')
print (results)
with open('results.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for r in results:
        writer.writerow([r])
# plt.bar(range(len(scores)), scores)
# plt.show()
# plt.bar(range(len(results)), results)
# plt.show()
