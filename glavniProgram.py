import gymnasium as gym
import numpy as np
import pickle
import graph_creator as gc
import q_learning as ql

#potrebno za registriranje okoline
gym.register(
    id="Taxi-10x10", # give it a unique id
    entry_point="bigger_taxi_env:BigTaxiEnv", # frozen_lake_enhanced = name of file 'frozen_lake_enhanced.py'
    max_episode_steps=700
)

def run_and_visualize(num_episodes,alpha, gamma, epsilon, epsilon_decay_rate, is_training=True, render=False, enhanced_map=False):
    rewards_per_episode, mean_rewards, num_successes, steps_per_episode, mean_steps = (
        ql.run(num_episodes, alpha, gamma, epsilon, epsilon_decay_rate, is_training, render, enhanced_map))

    ql.visualize(num_episodes, rewards_per_episode, mean_rewards, num_successes, steps_per_episode, mean_steps, a=alpha, g=gamma, e=epsilon, dr=epsilon_decay_rate)

def run_and_compare(num_episodes, alphas, gammas, epsilons, e_decay_rates, enhanced_map=False):
    all_rewards_per_episode = []
    all_mean_rewards_per_episode = []
    all_steps_per_episode = []
    all_mean_steps_per_episode = []
    all_num_successes = []

    for alpha, gamma, epsilon, e_decay_rate in zip(alphas, gammas, epsilons, e_decay_rates):
        rewards_per_episode, mean_rewards, num_successes, steps_per_episode, mean_steps = (
            ql.run(num_episodes, alpha, gamma, epsilon, e_decay_rate, enhanced_map=enhanced_map))

        all_rewards_per_episode.append(rewards_per_episode)
        all_mean_rewards_per_episode.append(mean_rewards)
        all_steps_per_episode.append(steps_per_episode)
        all_mean_steps_per_episode.append(mean_steps)
        all_num_successes.append(num_successes)

    gc.compare_succes_rate_graph(num_episodes, all_num_successes, alphas, gammas, epsilons, e_decay_rates)
    gc.compare_steps_mean_graph(num_episodes, all_steps_per_episode, all_mean_steps_per_episode, alphas, gammas, epsilons, e_decay_rates)
    gc.compare_mean_rewards_graph(num_episodes, all_rewards_per_episode, all_mean_rewards_per_episode, alphas, gammas, epsilons, e_decay_rates)


if __name__ == '__main__':
    # run_and_visualize(10000, 0.9,0.9, 1, 0.0001, enhanced_map=True)
    # run_and_visualize(10000, 0.9,0.9, 1, 0.0005, enhanced_map=True)
    # run_and_visualize(10000, 0.9,0.9, 1, 0.001, enhanced_map=True)
    # run_and_visualize(10000, 0.5,0.9, 1, 0.001, enhanced_map=True)
    run_and_visualize(10000, 0.7,0.9, 1, 0.001, enhanced_map=False)
    alphas = [0.9, 0.9, 0.7]
    gammas = [0.9, 0.9, 0.9]
    epsilons = [1, 1, 1]
    drs = [0.0001, 0.001, 0.001]
    # run_and_compare(10000, alphas, gammas, epsilons, drs, enhanced_map=True)

    ql.run(10, alpha=0.9, gamma=0.9, epsilon=1,epsilon_decay_rate=0.0001, is_training=False, render=True, enhanced_map=False)