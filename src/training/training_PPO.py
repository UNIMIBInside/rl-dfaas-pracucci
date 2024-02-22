import pathlib
import datetime
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys_path = 'C:/Users/giaco/Desktop/repos/RL-edge-computing/src' 
sys.path.append(sys_path)

import torch
from torch.distributions.dirichlet import Dirichlet
from torch.utils.tensorboard import SummaryWriter

def train_ppo_agent(env, agent, horizon=1024, epochs=10, num_episodes=20, max_steps_per_episode=100):
    # Save logs under 'logs/PPO' in the project root directory.
    # Each run has a different subdirectory based on start timestamp.
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = pathlib.Path(__file__).parent.parent.parent / 'logs' / 'PPO' / current_time
    print(f'Log folder: {str(log_path)!r}')

    # Checkpoints are saved under 'checkpoint/PPO' in the project root
    # directory.
    checkpoint_path = pathlib.Path(__file__).parent.parent.parent / 'checkpoints' / 'PPO'
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print(f'Checkpoint folder: {str(checkpoint_path)!r}')

    writer = SummaryWriter(log_path)

    total_rewards = []
    total_losses = []
    total_actor_losses = []
    total_critic_losses = []
    total_entropy_losses = []

    for episode in range(num_episodes):
        states = []
        actions = []
        rewards = []
        masks = []
        values = []
        old_probs = []
        state = env.reset()
        episode_reward = 0
        episode_losses = []
        episode_actor_losses = []
        episode_critic_losses = []
        episode_entropy_losses = []

        for step in range(max_steps_per_episode):
            #print("---------------------------------")
            print(f"Episode: {episode}, Step: {step}")
            #print("---------------------------------")

            action = agent.select_action(state)
            value = agent.critic(torch.FloatTensor(state).unsqueeze(0).to(agent.device)).item()
            next_state, reward, truncated, done, info = env.step(action)

            action_probs = agent.actor(torch.FloatTensor(state).unsqueeze(0).to(agent.device))
            dist = Dirichlet(action_probs)
            old_prob = dist.log_prob(torch.FloatTensor(action).to(agent.device)).item()

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            masks.append(1 - done)
            values.append(value)
            old_probs.append(old_prob)

            state = next_state
            episode_reward += reward

            if (step + 1) % horizon == 0 or done:
                loss, actor_loss, critic_loss, entropy = agent.update(states, actions, old_probs, rewards, masks, values, epochs=epochs)

                episode_losses.append(loss)
                episode_actor_losses.append(actor_loss)
                episode_critic_losses.append(critic_loss)
                episode_entropy_losses.append(entropy)

                states, actions, rewards, masks, values, old_probs = [], [], [], [], [], []

            if (episode + 1) % 50 == 0:  # Salva un checkpoint ogni 500 episodi
                checkpoint_file = checkpoint_path / f'checkpoint_{episode + 1}'
                # Torch doesn't support pathlib objects, it must be a string.
                agent.save_weights_PPO(str(checkpoint_file))

            if done:
                break

        # Average out the losses over the episode
        avg_loss = sum(episode_losses) / len(episode_losses)
        avg_actor_loss = sum(episode_actor_losses) / len(episode_actor_losses)
        avg_critic_loss = sum(episode_critic_losses) / len(episode_critic_losses)
        avg_entropy_loss = sum(episode_entropy_losses) / len(episode_entropy_losses)

        writer.add_scalar('Loss', avg_loss, episode)
        writer.add_scalar('Reward', episode_reward, episode)
        writer.add_scalar('Actor Loss', avg_actor_loss, episode)
        writer.add_scalar('Critic Loss', avg_critic_loss, episode)
        writer.add_scalar('Entropy Loss', avg_entropy_loss, episode)
        
        total_rewards.append(episode_reward)
        total_losses.append(avg_loss)
        total_actor_losses.append(avg_actor_loss)
        total_critic_losses.append(avg_critic_loss)
        total_entropy_losses.append(avg_entropy_loss)

        #print(f"Episode: {episode + 1}, Reward: {episode_reward}, Actor Loss: {avg_actor_loss}, Critic Loss: {avg_critic_loss}")

    writer.close()

    weights_file = checkpoint_path / 'PPO_weights'
    agent.save_weights_PPO(str(weights_file))

    # Plot total rewards
    #plt.figure(figsize=(12, 8))
    
    #plt.subplot(2, 2, 1)
    #plt.plot(total_rewards)
    #plt.xlabel('Episode')
    #plt.ylabel('Reward')
    #plt.title('Total Rewards')
    
    #plt.subplot(2, 2, 2)
    #plt.plot(total_actor_losses)
    #plt.xlabel('Episode')
    #plt.ylabel('Actor Loss')
    #plt.title('Actor Losses')
    
    #plt.subplot(2, 2, 3)
    #plt.plot(total_critic_losses)
    #plt.xlabel('Episode')
    #plt.ylabel('Critic Loss')
    #plt.title('Critic Losses')
    
    #plt.subplot(2, 2, 4)
    #plt.plot(total_entropy_losses)
    #plt.xlabel('Episode')
    #plt.ylabel('Entropy Loss')
    #plt.title('Entropy Losses')

    #plt.tight_layout()
    #plt.show()
    
    return total_rewards
