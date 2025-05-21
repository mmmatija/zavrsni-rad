import gymnasium as gym
import numpy as np
import pickle
import graph_creator as gc

def run(num_episodes,alpha, gamma, epsilon, epsilon_decay_rate, is_training=True, render=False, enhanced_map=False):
    q_table_file = f"taxi_q_table_{alpha}_{gamma}_{epsilon}_{epsilon_decay_rate}.pkl"
    #initializing the environment
    if enhanced_map:
        env = gym.make('Taxi-10x10', render_mode='human' if render else None)
    else:
        env = gym.make('Taxi-v3', render_mode='human' if render else None)
    #initializing/loading q-table
    if is_training:
        q_table = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open(q_table_file, 'rb')
        q_table = pickle.load(f)
        f.close()

    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(num_episodes)
    num_successes = np.zeros(num_episodes)
    steps_per_episode = np.zeros(num_episodes)

    #algorithm implementation
    for ep in range(num_episodes):
        #initializing environment for episode
        state, _ = env.reset()
        terminated = False
        truncated = False

        rewards=0
        steps=0
        while not terminated and not truncated:
            #decide which action to take
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])
            #do the action
            new_state, reward, terminated, truncated, _ = env.step(action)
            #Bellman equation - updating q-table
            if is_training:
                q_table[state, action] = (1-alpha)*q_table[state, action] + alpha*(reward + gamma * np.max(q_table[new_state,:]))

            rewards += reward
            steps+=1
            #changing to the new state
            state = new_state
        #decay of epsilon
        epsilon = max(0, epsilon - epsilon_decay_rate)
        if(epsilon==0):
            alpha = 0.01

        rewards_per_episode[ep] = rewards
        steps_per_episode[ep] = steps
        if terminated:
            num_successes[ep] = 1

    env.close()

    mean_rewards = np.zeros(num_episodes)
    mean_steps = np.zeros(num_episodes)
    for t in range(num_episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t - 100):(t + 1)])
        mean_steps[t] = np.mean(steps_per_episode[max(0, t - 100):(t + 1)])

    if is_training:
        f = open(q_table_file,"wb")
        pickle.dump(q_table, f)
        f.close()

    return rewards_per_episode, mean_rewards, num_successes, steps_per_episode, mean_steps

def visualize(num_episodes, rewards_per_episode, mean_rewards, num_successes, steps_per_episode, mean_steps, a, g, e,dr):
    gc.create_mean_reward_graph(num_episodes, rewards_per_episode, mean_rewards, a=a, g=g, e=e, dr=dr)
    gc.create_mean_steps_graph(num_episodes, steps_per_episode, mean_steps, a=a, g=g, e=e, dr=dr)
    gc.create_succes_rate_graph(num_episodes, num_successes, a=a, g=g, e=e, dr=dr)