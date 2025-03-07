

import matplotlib.pyplot as plt
import numpy as np




fig, axes = plt.subplots(2, 2, figsize=(10, 7))  # 1 row, 3 columns

# Plot data in each subplot
axes[0,0].plot(np.load("Q_learning_Maze_1_adaptive_eps_time.npy"), 
             np.load("Q_learning_maze_1_adaptive_eps_cum_rewards.npy"), color = 'g',label = 'Q learning, adaptive epsilon')
axes[0,0].plot(np.load("SARSA_Maze_1_adaptive_eps_time.npy"), 
                     np.load("SARSA_Maze_1_adaptive_eps_cum_rewards.npy"), color = 'g', label = 'SARSA, adaptive epsilon', linestyle = "--")
axes[0,0].grid(True)
axes[0,0].set_title("A", loc='left')
axes[0,0].set_xlabel("state transition")
axes[0,0].set_ylabel("cumulative rewards")
axes[0,0].legend(loc="upper left")

axes[0,1].plot(np.load("Q_learning_Maze_2_adaptive_eps_time.npy"), 
             np.load("Q_learning_maze_2_adaptive_eps_cum_rewards.npy"), color = 'g', label = 'Q learning, adaptive epsilon')
axes[0,1].plot(np.load("SARSA_Maze_2_adaptive_eps_time.npy"), 
                     np.load("SARSA_Maze_2_adaptive_eps_cum_rewards.npy"), color = 'g', label = 'SARSA, adaptive epsilon', linestyle = "--")
axes[0,1].grid(True)
axes[0,1].set_title("B", loc='left')
axes[0,1].set_xlabel("state transition")
axes[0,1].set_ylabel("cumulative rewards")
axes[0,1].legend(loc="upper left")

axes[1,0].plot(np.load("Q_learning_Maze_1_adaptive_eps_time.npy"), 
             np.load("Q_learning_maze_1_adaptive_eps_cum_rewards.npy"), color = 'brown',label = 'Adaptive epsilon')
axes[1,0].plot(np.load("Q_learning_Maze_1_Boltzmann_time.npy"), 
             np.load("Q_learning_Maze_1_Boltzmann_cum_rewards.npy"), color = 'purple',label = 'Boltzmann exploration')
axes[1,0].plot(np.load("Q_learning_Maze_1_eps_greedy_0.25_time.npy"), 
             np.load("Q_learning_maze_1_eps_greedy_0.25_cum_rewards.npy"), color = 'blue',label = 'epsilon-greedy = 0.25')
axes[1,0].plot(np.load("Q_learning_Maze_1_eps_greedy_0.5_time.npy"), 
             np.load("Q_learning_maze_1_eps_greedy_0.5_cum_rewards.npy"), color = 'green',label = 'epsilon-greedy = 0.5')
axes[1,0].plot(np.load("Q_learning_Maze_1_eps_greedy_0.75_time.npy"), 
             np.load("Q_learning_maze_1_eps_greedy_0.75_cum_rewards.npy"), color = 'orange',label = 'epsilon-greedy = 0.75')
axes[1,0].plot(np.load("Q_learning_Maze_1_eps_greedy_0.1_time.npy"), 
             np.load("Q_learning_maze_1_eps_greedy_0.1_cum_rewards.npy"), color = 'red',label = 'epsilon-greedy = 0.1')
axes[1,0].plot(np.load("Q_learning_Maze_1_eps_greedy_0.9_time.npy"), 
             np.load("Q_learning_maze_1_eps_greedy_0.9_cum_rewards.npy"), color = 'black',label = 'epsilon-greedy = 0.9')
axes[1,0].grid(True)
axes[1,0].set_title("C", loc='left')
axes[1,0].set_xlabel("state transition")
axes[1,0].set_ylabel("cumulative rewards")
axes[1,0].legend(loc="upper left")


axes[1,1].plot(np.load("Q_learning_Maze_2_adaptive_eps_time.npy"), 
             np.load("Q_learning_maze_2_adaptive_eps_cum_rewards.npy"), color = 'brown',label = 'adaptive epsilon')
axes[1,1].plot(np.load("Q_learning_Maze_2_Boltzmann_time.npy"), 
             np.load("Q_learning_Maze_2_Boltzmann_cum_rewards.npy"), color = 'purple',label = 'Boltzmann exploration')
axes[1,1].plot(np.load("Q_learning_Maze_2_eps_0.25_time.npy"), 
             np.load("Q_learning_maze_2_eps_0.25_cum_rewards.npy"), color = 'blue',label = 'epsilon-greedy = 0.25')
axes[1,1].plot(np.load("Q_learning_Maze_2_eps_0.5_time.npy"), 
             np.load("Q_learning_maze_2_eps_0.5_cum_rewards.npy"), color = 'green',label = 'epsilon-greedy = 0.5')
axes[1,1].plot(np.load("Q_learning_Maze_2_eps_0.75_time.npy"), 
             np.load("Q_learning_maze_2_eps_0.75_cum_rewards.npy"), color = 'orange',label = 'epsilon-greedy = 0.75')
axes[1,1].plot(np.load("Q_learning_Maze_2_eps_0.1_time.npy"), 
             np.load("Q_learning_maze_2_eps_0.1_cum_rewards.npy"), color = 'red',label = 'epsilon-greedy = 0.1')
axes[1,1].plot(np.load("Q_learning_Maze_2_eps_0.9_time.npy"), 
             np.load("Q_learning_maze_2_eps_0.9_cum_rewards.npy"), color = 'black',label = 'epsilon-greedy 0.9')

             
axes[1,1].grid(True)
axes[1,1].grid(True)
axes[1,1].set_title("D", loc='left')
axes[1,1].set_xlabel("state transition")
axes[1,1].set_ylabel("cumulative rewards")
axes[1,1].legend(loc="upper left")



plt.tight_layout()
plt.show()