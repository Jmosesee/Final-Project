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

# exploration_rate = 1.0
# exploration_min = 0.01
# exploration_decay = 0.995
# sample_batch_size = 32
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Todo: Try the model from D
def build_model():
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation ='relu'))  #Todo: Try 16, 12
    model.add(Dense(48, activation ='relu'))                        #Todo: Try 16, 12
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    return model

# targets = []
def decide_action(state, model):
    # if np.random.rand() <= exploration_rate:
    #     # The agent acts randomly
    #     return env.action_space.sample()
    # Predict the reward value based on the given state
    act_values = model.predict(state)
    # Pick the action based on the predicted reward
    return np.argmax(act_values[0])

# Run one episode
def run_episode(model):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    index = 0
    episode_memory = deque(maxlen=2000)

    while not done:
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
    reward = -1
    while True:
        try:
            (state, action, _, next_state, _) = episode_memory.pop()
        except:
            break
        # target = reward
        # if not done:
        #     prediction = model.predict(next_state)
        #     print(prediction)
        #     target = reward + gamma * np.amax(prediction[0])
        target = reward
        target_f = model.predict(state)
        # print (target)
        # targets.append(target)
        # target_f[0][action] = gamma * target_f[0][action] + target
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
        reward *= gamma

# Test the agent for episodes
def test_agent(episodes):
    total_score = 0
    model = build_model()
    # model.reset_states()
    scores = []
    last_scores = deque(maxlen=100)
    for e in range(episodes):
        # state = env.reset()
        # state = np.reshape(state, [1, state_size])
        # done = False
        # index = 0
        # episode_memory = deque(maxlen=2000)

        # Run one episode
        # while not done:
        #     # env.render()
        #     action = decide_action(state)
        #     next_state, reward, done, _=env.step(action)  # Execute action
        #     next_state = np.reshape(next_state, [1, state_size])
        #     episode_memory.append((state, action, reward, next_state, done))
        #     state = next_state
        #     index += 1
        #     if (reward != 1):   # This never happens
        #         print(f"Episode {e}# Score:{reward}")
        (score, episode_memory) = run_episode(model)
        total_score += score
        scores.append(score)
        last_scores.append(score)
        print(f"Episode {e}# Score:{score}")
        with open('log.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow((e, score))
        #   Train to avoid these mistakes
        if score<500:
            train_agent(episode_memory, model)
            # reward = -1
            # while True:
            #     try:
            #         (state, action, _, next_state, _) = episode_memory.pop()
            #     except:
            #         break
            #     # target = reward
            #     # if not done:
            #     #     prediction = model.predict(next_state)
            #     #     print(prediction)
            #     #     target = reward + gamma * np.amax(prediction[0])
            #     target = reward
            #     target_f = model.predict(state)
            #     # print (target)
            #     # targets.append(target)
            #     # target_f[0][action] = gamma * target_f[0][action] + target
            #     target_f[0][action] = target
            #     model.fit(state, target_f, epochs=1, verbose=0)
            #     reward *= gamma
        # if exploration_rate > exploration_min:
        #     exploration_rate *= exploration_decay

        # Report result
        mean_score = np.mean(last_scores)
        if mean_score >= n_win_score and e >= 100:
            print('Ran {} episodes. Solved after {} episodes ✔ Total score: {}'.format(e, e - 100, total_score))
            break;
        # if np.sum(last_scores)/20 > 495:
        #     break
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
