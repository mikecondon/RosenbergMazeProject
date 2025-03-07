import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx


def Maze_tree_back_forward_branching():
    children = {}
    parent = {}
    depth = {}
    
    root = "R0"
    level_1 = ["R", "L"]
    children[root] = level_1
    depth["R0"] = 0
    for node in level_1:
        parent[node] = "R0"
        depth[node] = 1
        
    def add_level(current_nodes, depth_transition):
        next_nodes = []
        for node in current_nodes:
            children[node] = []
            for i in level_1:
                new_node = f"{node}{i}"
                children[node].append(new_node)
                parent[new_node] = node
                depth[new_node] = depth_transition
                next_nodes.append(new_node)
        return next_nodes
    
    level_2 = add_level(level_1, 2)
    level_3 = add_level(level_2, 3)
    level_4 = add_level(level_3, 4)
    level_5 = add_level(level_4, 5)
    level_6 = add_level(level_5, 6)
    
    
    return children, parent, depth

children, parent, depth = Maze_tree_back_forward_branching()


##Define starting and goal state
start_state = "R0"
goal_state = "RLLRLR" ##


# Define the Transition Function
def move_to_next_state(state, action):
    if action == "back":
        return parent.get(state, state)
    # Check if state has children
    if state in children and children[state]:
        mapping = {"left": 0, "right": 1}
        if action in mapping and mapping[action] < len(children[state]):
            return children[state][mapping[action]]
        else:
            return state
    return state

def get_reward(state, next_state):
    if next_state == goal_state:
        return 5.0
    else:
        return 0.0


# Q-Learning Setup
all_nodes = list(depth.keys())
ACTIONS = ["back", "left", "right"]
Q = {s: {a: 0.0 for a in ACTIONS} for s in all_nodes if s != goal_state}
Q[goal_state] = {}

#appropriate_actions = {}
#appropriate_actions["R"] = None

def adaptive_max_ent_epsilon_greedy(state, Q_table, base_epsilon=0.1, temperature=0.5, entropy_weight=0.7):
    q_vals = np.array(list(Q_table[state].values()))
    q_vals_norm = q_vals - np.max(q_vals)
    exp_q = np.exp(q_vals_norm / temperature)
    probs = exp_q / np.sum(exp_q)
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    max_entropy = np.log(len(q_vals))
    normalized_entropy = entropy / max_entropy
    adapted_epsilon = base_epsilon + entropy_weight * normalized_entropy
    return max(0.0, min(1.0, adapted_epsilon))

def epsilon_greedy(epsilon):
    return epsilon
    
def select_action(state):
    if state == goal_state:
        return None
    eps = adaptive_max_ent_epsilon_greedy(state, Q)
    #eps = epsilon_greedy(0.1)
    if random.random() < eps:
        return random.choice(ACTIONS)
    else:
        state_actions = Q[state]
        max_val = max(state_actions.values())
        best = [a for a, v in state_actions.items() if v == max_val]
        return random.choice(best)

def Boltzmann_exploration_policy(state, Q_table, temperature=0.5):
    state_actions = list(Q_table[state].keys())  
    q_action_vals = np.array(list(Q_table[state].values()))  
    if temperature <= 1e-5:
        return random.choice(state_actions)
    q_vals_action_norm = q_action_vals - np.max(q_action_vals) 
    exp_q = np.exp(q_vals_action_norm / temperature)
    softmax = exp_q / np.sum(exp_q)
    return np.random.choice(state_actions, p=softmax)


# Q-Learning Simulation

#initialize parameters

alpha = 0.1
gamma = 1
num_steps = 40000

random.seed(42)
np.random.seed(42)
cumulative_goal_reaches = []
time_steps_record = []
goal_count = 0
current_state = start_state
direct_path = []
cumulative_direct_paths = []
iterator = 0
direct_path_time = []
correct_direct_path = ['R0', 'R', 'RL', 'RLL', 'RLLR', 'RLLRL', 'RLLRLR']
direct_path = []        
direct_path_time = []

current_state = start_state

for t in range(1, num_steps+1):
    action = select_action(current_state)
    #action = Boltzmann_exploration_policy(current_state, Q, temperature=0.5)
    if action is None:
        current_state = start_state
        time_steps_record.append(t)
        cumulative_goal_reaches.append(goal_count)
        direct_path = []
        continue

    next_st = move_to_next_state(current_state, action)
    r = get_reward(current_state, next_st)
    
    # Q-learning update
    if next_st == goal_state:
        Q[current_state][action] += alpha * (r - Q[current_state][action])
    else:
        best_next = max(Q[next_st].values()) if Q[next_st] else 0.0
        Q[current_state][action] += alpha * (r + gamma * best_next - Q[current_state][action])
        
    expected = None
    if not direct_path:
        expected = "R0"
    else:
        index = len(direct_path)
        if index < len(correct_direct_path):
            expected = correct_direct_path[index]
    
    if next_st == expected:
        direct_path.append(next_st)
    else:
        direct_path = []
    
    
    if next_st == goal_state:
        if direct_path == correct_direct_path:
            direct_path_time.append(t)
        goal_count += 1
        current_state = start_state
        
        direct_path = []
    else:
        current_state = next_st

    time_steps_record.append(t)
    cumulative_goal_reaches.append(goal_count)

print([cumulative_goal_reaches])

plt.figure(figsize=(8, 5))
plt.plot(time_steps_record, cumulative_goal_reaches, '-', linewidth=2, color='green', label="Goal Reaches")
plt.plot(direct_path_time, [i for i in range(1, len(direct_path_time)+1)], '-', linewidth=2, color='red', label="Direct Path Completions")
plt.title("Cumulative Goal Reaches and Direct Path Completions\n(Maze: Start at G323, Goal at S312)")
plt.xlabel("Time Step")
plt.ylabel("Cumulative Count")
plt.legend()
plt.grid(True)
plt.show()

x_val = np.save("Q_learning_Maze_1_adaptive_eps_time.npy",time_steps_record)
y_vals = np.save("Q_learning_maze_1_adaptive_eps_cum_rewards.npy",cumulative_goal_reaches)

'''
x_val = np.save("Q_learning_Maze_1_Boltzmann_time.npy",time_steps_record)
y_vals = np.save("Q_learning_maze_1_Boltzmann_cum_rewards.npy",cumulative_goal_reaches)
'''
'''
x_val = np.save("Q_learning_Maze_1_eps_greedy_0.25_time.npy",time_steps_record)
y_vals = np.save("Q_learning_maze_1_eps_greedy_0.25_rewards.npy",cumulative_goal_reaches)
'''
'''
x_val = np.save("Q_learning_Maze_1_eps_greedy_0.75_time.npy",time_steps_record)
y_vals = np.save("Q_learning_maze_1_eps_greedy_0.75_cum_rewards.npy",cumulative_goal_reaches)
'''
'''
x_val = np.save("Q_learning_Maze_1_eps_greedy_0.5_time.npy",time_steps_record)
y_vals = np.save("Q_learning_maze_1_eps_greedy_0.5_cum_rewards.npy",cumulative_goal_reaches)
'''
'''
x_val = np.save("Q_learning_Maze_1_eps_greedy_0.9_time.npy",time_steps_record)
y_vals = np.save("Q_learning_maze_1_eps_greedy_0.9_cum_rewards.npy",cumulative_goal_reaches)
'''
'''
x_val = np.save("Q_learning_Maze_1_eps_greedy_0.1_time.npy",time_steps_record)
y_vals = np.save("Q_learning_maze_1_eps_greedy_0.1_cum_rewards.npy",cumulative_goal_reaches)
'''