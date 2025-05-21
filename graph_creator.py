import numpy as np
import matplotlib.pyplot as plt
import os

def create_mean_reward_graph(num_episodes, rewards_per_episode, mean_rewards, a, g, e, dr):
    plt.figure(figsize=(12, 6))
    min_rewards = np.zeros(num_episodes)
    max_rewards = np.zeros(num_episodes)
    for i in range(num_episodes):
        min_rewards[i] = np.min(rewards_per_episode[max(0, i - 100):(i + 1)])
        max_rewards[i] = np.max(rewards_per_episode[max(0, i - 100):(i + 1)])

    plt.plot(mean_rewards, label=f"α={a} γ={g}\nε₀={e} δ={dr}\n")
    plt.fill_between(range(num_episodes), min_rewards, max_rewards, alpha=0.2)
    plt.ylabel('reward mean of 100 eps')
    plt.xlabel('episode')
    plt.title('Mean rewards per episode')
    plt.legend()
    directory = f"taxi_{a}_{g}_{e}_{dr}"
    os.makedirs(directory, exist_ok=True)
    fullpath = os.path.join(directory, f"taxi_rewards_mean.png")
    plt.savefig(fullpath)
    plt.close()



def create_mean_steps_graph(num_episodes, steps_per_episode, mean_steps, a, g, e, dr):
    min_steps = np.zeros(num_episodes)
    max_steps = np.zeros(num_episodes)
    for i in range(num_episodes):
        min_steps[i] = np.min(steps_per_episode[max(0, i - 100):(i + 1)])
        max_steps[i] = np.max(steps_per_episode[max(0, i - 100):(i + 1)])

    plt.figure(figsize=(12, 6))
    plt.plot(mean_steps, label=f"α={a} γ={g}\nε₀={e} δ={dr}\n")
    plt.fill_between(range(num_episodes), min_steps, max_steps, alpha=0.2)
    plt.xticks(range(0,num_episodes+1,int(num_episodes*0.1)))
    plt.ylabel('step mean of 100 eps')
    plt.xlabel('episode')
    plt.title('Mean steps per episode')
    directory = f"taxi_{a}_{g}_{e}_{dr}"
    os.makedirs(directory, exist_ok=True)
    fullpath = os.path.join(directory, f"taxi_steps_mean.png")
    plt.savefig(fullpath)
    plt.close()

def create_succes_rate_graph(num_episodes, num_successes, a, g, e, dr):
    plt.figure(figsize=(12, 6))
    plt.scatter(range(num_episodes), num_successes, s=1, color='red', label='Success')
    plt.plot(np.cumsum(num_successes) / (np.arange(num_episodes) + 1), label=f"α={a} γ={g}\nε₀={e} δ={dr}\n")
    plt.ylabel('success rate')
    plt.xlabel('episode')
    plt.title('Success Rate')
    plt.legend()
    directory = f"taxi_{a}_{g}_{e}_{dr}"
    os.makedirs(directory, exist_ok=True)
    fullpath = os.path.join(directory, f"taxi_success_rate.png")
    plt.savefig(fullpath)
    plt.close()

def compare_succes_rate_graph(num_episodes, num_successes, alphas, gammas, epsilons, decay_rates):
    plt.figure(figsize=(12, 6))
    for (successes, a,g,e,dr) in zip(num_successes, alphas, gammas, epsilons, decay_rates):
        plt.plot(np.cumsum(successes) / (np.arange(num_episodes) + 1), label=f"α={a} γ={g}\nε₀={e} δ={dr}\n")
    plt.ylabel('success rate')
    plt.xlabel('episode')
    plt.title('Success Rate Comparison')
    plt.legend()
    plt.savefig('taxi_success_rate_comparison.png')
    plt.close()

def compare_steps_mean_graph(num_episodes, steps_per_episode, mean_steps, alphas, gammas, epsilons, decay_rates):
    plt.figure(figsize=(12, 6))
    for i, (spe, me, a,g,e,dr) in enumerate(zip(steps_per_episode, mean_steps, alphas, gammas, epsilons, decay_rates)):
        min_steps = np.zeros(num_episodes)
        max_steps = np.zeros(num_episodes)
        for i in range(num_episodes):
            min_steps[i] = np.min(spe[max(0, i - 100):(i + 1)])
            max_steps[i] = np.max(spe[max(0, i - 100):(i + 1)])
        plt.plot(me, label=f"{i}")
    plt.ylabel('steps per episode')
    plt.xlabel('episode')
    plt.title('Mean steps per episode Comparison')
    plt.legend()
    plt.savefig('taxi_steps_mean_comparison.png')
    plt.close()

def compare_mean_rewards_graph(num_episodes, rewards_per_episode, mean_rewards, alphas, gammas, epsilons, decay_rates):
    plt.figure(figsize=(12, 6))
    for i, (rpe, mr, a,g,e,dr) in enumerate(zip(rewards_per_episode, mean_rewards, alphas, gammas, epsilons, decay_rates)):
        min_steps = np.zeros(num_episodes)
        max_steps = np.zeros(num_episodes)
        for i in range(num_episodes):
            min_steps[i] = np.min(rpe[max(0, i - 100):(i + 1)])
            max_steps[i] = np.max(rpe[max(0, i - 100):(i + 1)])
        plt.plot(mr, label=f"{i}")
    plt.ylabel('rewards per episode')
    plt.xlabel('episode')
    plt.title('Mean rewards per episode Comparison')
    plt.legend()
    plt.savefig('taxi_rewards_mean_comparison.png')
    plt.close()