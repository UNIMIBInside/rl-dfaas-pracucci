import torch
import sys
sys.path.append('C:/Users/giaco/Desktop/tesi_git/src')
from env.env import TrafficManagementEnv
from SAC.SAC import SAC
import matplotlib.pyplot as plt

state_dim = 4  
action_dim = 3  
agent = SAC(state_dim, action_dim, device=torch.device("cpu"))

path_to_weights = "C:/Users/giaco/Desktop/local-git/SAC_weights/SAC_weights"  
agent.load_weights_SAC(path_to_weights)

env = TrafficManagementEnv()  
num_episodes = 50
all_episode_rewards = []
all_episode_rejections = []
all_managed_requests_per_episode = []

prev_total_requests = 0

for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        episode_reward += reward
        state = next_state
    print(f"Episodio {episode + 1}: Ricompensa Totale = {episode_reward}")
    all_episode_rewards.append(episode_reward)
    all_episode_rejections.append(env.total_rejected_requests)
    # Calcolo delle richieste gestite per l'episodio corrente
    managed_requests_this_episode = env.total_managed_requests - prev_total_requests
    all_managed_requests_per_episode.append(managed_requests_this_episode)
    # Aggiornamento per il prossimo ciclo
    prev_total_requests = env.total_managed_requests
# Calcola la percentuale di richieste rifiutate per ogni episodio
rejection_percentages = [(rejections/requests) * 100 if requests != 0 else 0 for rejections, requests in zip(all_episode_rejections, all_managed_requests_per_episode)]

# Plotting
plt.figure(figsize=(18,5))

# subplot per la ricompensa
plt.subplot(1, 3, 1)
plt.plot(all_episode_rewards, marker='o', linestyle='-')
plt.title('Ricompensa Totale per Episodio')
plt.xlabel('Episodi')
plt.ylabel('Ricompensa Totale')
plt.grid(True)

# subplot per la percentuale di rifiuti
plt.subplot(1, 3, 2)
plt.plot(rejection_percentages, marker='o', linestyle='-', color='blue')
plt.title('Percentuale di Richieste Rifiutate per Episodio')
plt.xlabel('Episodi')
plt.ylabel('Percentuale di Richieste Rifiutate')
plt.ylim(0, 100) 
plt.grid(True)

# subplot per il numero assoluto di rifiuti e il numero totale di richieste gestite
plt.subplot(1, 3, 3)
plt.plot(all_episode_rejections, marker='o', linestyle='-', color='red', label='Richieste Rifiutate')
plt.plot(all_managed_requests_per_episode, marker='o', linestyle='--', color='green', label='Richieste Gestite')
plt.title('Numero di Richieste per Episodio')
plt.xlabel('Episodi')
plt.ylabel('Numero di Richieste')
plt.legend(loc='upper left')  
plt.grid(True)

plt.tight_layout()  
plt.show()
