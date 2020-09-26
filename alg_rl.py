# import gym
import keras
import numpy as np
import random
from os import remove

# from gym import wrappers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

import tensorflow as tf

import snake_engine_l

from collections import deque

ACTIONS_DIM = 3
OBSERVATIONS_DIM = 450
MAX_ITERATIONS = 10**3
LEARNING_RATE = 0.001
ALPHA = 1  #  renew outputs

NUM_EPOCHS = 5

GAMMA = 0.90
REPLAY_MEMORY_SIZE = 2048
NUM_EPISODES = 10000
TARGET_UPDATE_FREQ = 100
MINIBATCH_SIZE = 1028

RANDOM_ACTION_DECAY = 0.99
INITIAL_RANDOM_ACTION = 1
BOLZTMAN = False


class ReplayBuffer():

    def __init__(self, max_size):
        self.max_size = max_size
        self.transitions = deque()

    def add(self, observation, action, reward, observation2):
        if len(self.transitions) > self.max_size:
            self.transitions.popleft()
        self.transitions.append((observation, action, reward, observation2))

    def sample(self, count):
        # samples = deque(list(self.transitions)[len(self.transitions)-count:])
        # self.transitions = deque(list(self.transitions)[:len(self.transitions)-count], self.max_size)
        # return random.sample(samples, count)

        return random.sample(self.transitions, count)

    def size(self):
        return len(self.transitions)


def train(model, observations, targets):
    # for i, observation in enumerate(observations):
    #   np_obs = np.reshape(observation, [-1, OBSERVATIONS_DIM])
    #   print "t: {}, p: {}".format(model.predict(np_obs),targets[i])
    # exit(0)

    np_obs = np.reshape(observations, [-1, OBSERVATIONS_DIM, ])
    np_targets = np.reshape(targets, [-1, ACTIONS_DIM])

    acc = model.fit(np_obs, np_targets, epochs=NUM_EPOCHS, verbose=0, batch_size=128)
    acc = int(acc.history['accuracy'][-1]*100)/100
    return acc

# take action to move
def predict(model, observation):
    np_obs = np.reshape(observation, [-1, OBSERVATIONS_DIM, ])
    if BOLZTMAN:
        action = boltzman_choise(model.predict(np_obs))
    else:
        action = np.argmax(model.predict(np_obs))
    return action

# take outputs for training
def get_q(model, observation):
    np_obs = np.reshape(observation, [-1, OBSERVATIONS_DIM, ])
    return model.predict(np_obs)

def get_model():
    model = Sequential()
    # model.add(Convolution1D(2, (3), activation='relu'))
    # model.add(MaxPool1D(3))
    # model.add(Flatten())
    model.add(Dense(100, input_shape=(OBSERVATIONS_DIM,), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(ACTIONS_DIM, activation='linear'))

    model.compile(
        optimizer=Adam(lr=LEARNING_RATE),
        loss='mse',
        metrics=['accuracy'],
    )
    # print(model.summary())

    return model

def get_model2():
    input1 = tf.keras.Input(shape=(OBSERVATIONS_DIM,), name='map')
    input2 = tf.keras.Input(shape=(4,), name='options')
    # layer1 = tf.keras.layers.Convolution2D(1, (3, 3), padding='same', strides=(1, 1), activation='relu')(input1)
    # layer2 = tf.keras.layers.MaxPooling2D((2, 2))(layer1)
    # layer2 = tf.keras.layers.Dropout(0.25)(layer2)
    # layer3 = tf.keras.layers.Conv2D(2, (3, 3), padding='same', activation='relu')(layer2)
    # layer4 = tf.keras.layers.MaxPooling2D((2, 2))(layer3)
    # layer4 = tf.keras.layers.Dropout(0.25)(layer4)
    # layer5 = tf.keras.layers.Conv2D(4, (3, 3), padding='same', activation='relu')(layer4)
    # layer6 = tf.keras.layers.MaxPooling2D((2, 2))(layer5)
    # layer6 = tf.keras.layers.Dropout(0.25)(layer6)
    # layer6 = tf.keras.layers.Conv2D(16, (3,3), padding='same',activation='relu')(layer6)
    # layer6 = tf.keras.layers.MaxPooling2D((2,2))(layer6)

    # layer7 = tf.keras.layers.Flatten()(layer6)
    # layer8 = tf.keras.layers.concatenate([layer7, input2], axis=1)
    # layer8 = tf.expand_dims(layer8, 2)
    # layer9 = tf.keras.layers.LSTM(units=20)(layer8)

    layer10 = tf.keras.layers.Dense(100, activation='relu')(input1)
    layer10 = tf.keras.layers.Dropout(0.3)(layer10)
    layer11 = tf.keras.layers.Dense(50, activation='relu')(layer10)

    output1 = tf.keras.layers.Dense(ACTIONS_DIM, activation='linear')(layer11)

    model = tf.keras.models.Model(inputs=input1, outputs=output1, name='Simple')

    opt = tf.keras.optimizers.Adam(LEARNING_RATE)
    model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

    bolztman = tf.keras.layers.Softmax()
    # bolztman.compile()

    return model, bolztman

def update_action(action_model, sample_transitions):
    random.shuffle(sample_transitions)
    batch_observations = []
    batch_targets = []

    for sample_transition in sample_transitions:
        old_observation, action, reward, observation = sample_transition

        targets = get_q(action_model, old_observation)[0]
        targets[action] = reward*ALPHA + targets[action]*(1 - ALPHA)
        if observation is not None:
            predictions = get_q(action_model, observation)
            new_action = np.argmax(predictions)
            targets[action] += ALPHA * GAMMA * predictions[0, new_action]

        batch_observations.append(old_observation)
        batch_targets.append(targets)

    return train(action_model, batch_observations, batch_targets)

# choise action by boltzman propability
def boltzman_choise(output):
    # boltzmann choise
    # print('O', output, end=' ')
    expp = np.exp(output[0])
    expp /= expp.sum()
    # print('B', expp, end='| ')
    choise = random.random()
    bolt_sum = 0
    for i, o in enumerate(expp):
        bolt_sum += o
        if bolt_sum >= choise:
            # print(np.argmax(output), i)
            return i
    return np.argmax(output)



def main():
    steps_until_reset = TARGET_UPDATE_FREQ
    random_action_probability = INITIAL_RANDOM_ACTION

    data = list()
    accuracy = 0

    # Initialize replay memory D to capacity N
    replay = ReplayBuffer(REPLAY_MEMORY_SIZE)

    # Initialize action-value model with random weights
    action_model = get_model()

    try:
        action_model.load_weights('300alg_weights')
        print('Weights loaded')
    except:
        print('Weights not loaded')

    # Initialize target model with same weights
    #target_model = get_model()
    #target_model.set_weights(action_model.get_weights())

    # sn = snake_engine.Snake(4, 4)
    # env = snake_engine.game(15, 15, [sn])
    env = snake_engine_l.game(15, 15)
    # env.num_apples = 1
    last_iter = 0

    for episode in range(NUM_EPISODES):

        observation = env.reset()

        # oo = action_model.predict({'map': np.array(observation).reshape(1, 450), 'options': np.zeros((1,4))})
        # print(oo)
        # print(bolzman(oo))
        # print(boltzman_choise(bolzman, oo))
        # quit()

        for iteration in range(MAX_ITERATIONS):
            random_action_probability *= RANDOM_ACTION_DECAY
            random_action_probability = max(random_action_probability, 0.1)
            old_observation = observation

            # if episode % 10 == 0:
            #   env.render()

            if np.random.random() < random_action_probability:
                action = np.random.choice(range(ACTIONS_DIM))
            else:
                action = predict(action_model, observation)


            observation, reward, done = env.step(action)

            if iteration == MAX_ITERATIONS - 1:
                done = True


            if done:
                data.append([episode, iteration, 0, round(env.snake.score, 3), 0])
                mean_it = np.array(data[-50:])[:, 1].mean()
                mean_sc = np.array(data[-50:])[:, 3].mean()
                data[-1][2] = mean_it
                data[-1][4] = mean_sc
                _it = round(mean_it, 3)
                _sc = round(mean_sc, 3)
                print('Episode: {d[0]}, iterations: {d[1]} ({it}), Score: {d[3]} ({sc}) Acc: {acc}'.format(
                    d=data[-1], it=_it, sc=_sc, acc=accuracy))

                # print action_model.get_weights()
                # print target_model.get_weights()

                #print 'Game finished after {} iterations'.format(iteration)
                # reward = -10
                replay.add(old_observation, action, reward, None)
                break

            replay.add(old_observation, action, reward, observation)

            # if replay.size() >= MINIBATCH_SIZE:
            #     sample_transitions = replay.sample(MINIBATCH_SIZE)
            #     accuracy = update_action(action_model, action_model, sample_transitions)
            #     steps_until_reset -= 1


        if (episode+1) % 500 == 0:
            np.savetxt('train_{}.csv'.format(episode+1), data, delimiter=',')
            xx = episode+1-500
            if xx > 0:
                remove('train_{}.csv'.format(xx))

        elif (episode+1) % 100 == 0:
            action_model.save_weights('ragil_300alg', overwrite=True)


        if iteration > 250 and last_iter*0.8 < iteration:
            last_iter = iteration
            action_model.save_weights('300alg_weights', overwrite=True)
            # if steps_until_reset == 0:
            #   target_model.set_weights(action_model.get_weights())
            #   steps_until_reset = TARGET_UPDATE_FREQ

        # if iteration > 600:
        #     global ALPHA, BOLZTMAN
        #     ALPHA = 0.3
            # BOLZTMAN = True


        # trianing
        sample_transitions = replay.sample(min(MINIBATCH_SIZE, replay.size()))
        accuracy = update_action(action_model, sample_transitions)

    action_model.save_weights('ragil_300alg')

if __name__ == "__main__":
    main()