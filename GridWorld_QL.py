import numpy as np


class GridWorld:
    def __init__(self, rows, cols, rewards, gamma, noise):
        self.rows = rows
        self.cols = cols
        self.rewards = rewards
        self.gamma = gamma
        self.noise = noise
        self.V = np.zeros((self.rows, self.cols))
        self.V[0, 3] = 1
        self.V[1, 3] = -1
        self.actions = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # Right, Left, Up, Down

    def value_iteration(self, max_iterations):
        for iteration in range(max_iterations):
            new_V = np.copy(self.V)
            delta = 0
            for i in range(self.rows):
                for j in range(self.cols):
                    if (i, j) in self.rewards:
                        continue  # Skip grid with rewards
                    if (i, j) == (1, 1):
                        continue  # Skip wall
                    actions_val = []
                    for actionA in self.actions:
                        next_val = 0
                        for action in self.actions:
                            next_i, next_j = i + action[0], j + action[1]
                            if not (0 <= next_i < self.rows and 0 <= next_j < self.cols):
                                continue  # Skip if next move is out of bounds

                            if action == actionA:
                                next_val += (1 - self.noise) * (self.gamma * self.V[next_i, next_j])
                            else:
                                next_val += (self.noise / 2) * (self.gamma * self.V[next_i, next_j])
                        actions_val.append(next_val)

                    max_val = max(actions_val)
                    delta = max(delta, np.abs(max_val - self.V[i, j]))  # Update delta for convergence check
                    new_V[i, j] = max_val
            self.V = new_V
            print("Iteration:", iteration + 1)
            self.print_grid_values()

            if delta < 1e-4:  # Convergence check
                print("Converged at iteration", iteration + 1)
                break

    def print_grid_values(self):
        for i in range(self.rows):
            row_str = "| "
            for j in range(self.cols):
                if (i, j) in self.rewards:
                    row_str += "{:+}".format(self.rewards[(i, j)])
                elif (i, j) == (1, 1):
                    row_str += "WALL"
                else:
                    row_str += "{:.3f}".format(self.V[i, j])
                row_str += " | "
            print(row_str)
        print("\n")

    def extract_policy(self, V):
        policy = np.empty((self.rows, self.cols), dtype=str)
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) in self.rewards:
                    policy[i, j] = "{:+}".format(self.rewards[(i, j)])
                    continue
                if (i, j) == (1, 1):
                    policy[i, j] = "WALL"
                    continue

                next_vals = []
                for action in self.actions:
                    next_i, next_j = i + action[0], j + action[1]
                    if not (0 <= next_i < self.rows and 0 <= next_j < self.cols):
                        next_vals.append(float("-inf"))
                    else:
                        next_vals.append(V[next_i, next_j])
                best_action = np.argmax(next_vals)

                if best_action == 0:
                    policy[i, j] = "Right"
                elif best_action == 1:
                    policy[i, j] = "Left"
                elif best_action == 2:
                    policy[i, j] = "Up"
                elif best_action == 3:
                    policy[i, j] = "Down"
        return policy


# Define the environment parameters
rows = 3
cols = 4
rewards = {(0, 3): 1, (1, 3): -1}
gamma = 0.9
noise = 0.2
max_iterations = 100

# Create and run the environment
env = GridWorld(rows, cols, rewards, gamma, noise)
env.value_iteration(max_iterations)

# Extract and print the policy
print("The extracted policy:")
policy = env.extract_policy(env.V)
for i in range(rows):
    row_str = "| "
    for j in range(cols):
        row_str += policy[i, j] + " | "
    print(row_str)


