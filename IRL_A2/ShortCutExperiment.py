# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
import numpy as np
import matplotlib.pyplot as plt
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent, nStepSARSAAgent


def run_repetitions(env: ShortcutEnvironment=None, agent_type="QLearning", n_rep=100, n_episodes=1000, epsilon=0.1, alpha=0.1, gamma=1.0):
    print(f"Running {agent_type} in {env.__class__.__name__} with: epsilon={epsilon}, alpha={alpha} and n_episodes={n_episodes}")
    all_return = []
    
    for _ in range(n_rep):
        if agent_type == "QLearning":
            agent = QLearningAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=epsilon, alpha=alpha, gamma=gamma, env=env)
        elif agent_type == "SARSA":
            agent = SARSAAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=epsilon, alpha=alpha, gamma=gamma, env=env)
        elif agent_type == "ExpectedSARSA":
            continue
        elif agent_type == "nStepSARSA":
            continue
        returns = agent.train(n_episodes)
        all_return.append(returns)

    all_return = np.array(all_return)
    # print(all_return.shape)  # (n_rep, n_episodes)
    # print(all_return)
    mean_returns = np.mean(all_return, axis=0)  # axis=0 : mean per episode - axis=1 : mean per repetition
    # print(mean_returns.shape)  # (n_episodes,)
    # print(mean_returns)
    
    return mean_returns


def plot_greedy_policy_with_paths(Q, env, title="Greedy Policy"):
    """
    Draws a matplotlib grid showing:
      - An arrow in every cell indicating the greedy action (argmax Q)
      - Cliff cells (C), Goal cell (G), and walls marked clearly
      - The greedy path traced from each of the two starting positions
    """
    r, c = env.r, env.c  # 12x12

    # Action encoding matches env.step(): 0=up, 1=down, 2=left, 3=right
    arrow_map = {0: (0, 0.3),   # up:    draw arrow pointing up
                 1: (0, -0.3),  # down:  draw arrow pointing down
                 2: (-0.3, 0),  # left
                 3: (0.3, 0)}   # right

    greedy_actions = np.argmax(Q, axis=1)  # best action per state, shape (144,)
    q_max = np.max(Q, axis=1)              # to detect unvisited states (all zeros)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(0, c)
    ax.set_ylim(0, r)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14)

    # ── Step 1: colour the background cells ─────────────────────────────────
    for y in range(r):
        for x in range(c):
            cell = env.s[y, x]
            if cell == 'C':
                # Cliff cells in red
                ax.add_patch(plt.Rectangle((x, r-1-y), 1, 1,
                             color='salmon', zorder=0))
            elif cell == 'G':
                # Goal cell in green
                ax.add_patch(plt.Rectangle((x, r-1-y), 1, 1,
                             color='lightgreen', zorder=0))

    # ── Step 2: draw an arrow for every non-cliff, non-goal state ────────────
    for state in range(r * c):
        x = state % c
        y = state // c
        # Flip y so row 0 is at the top, matching the printed output
        plot_y = r - 1 - y

        cell = env.s[y, x]
        if cell == 'C':
            ax.text(x + 0.5, plot_y + 0.5, 'C',
                    ha='center', va='center', fontsize=8,
                    color='darkred', fontweight='bold')
            continue
        if cell == 'G':
            ax.text(x + 0.5, plot_y + 0.5, 'G',
                    ha='center', va='center', fontsize=10,
                    color='darkgreen', fontweight='bold')
            continue

        if q_max[state] == 0:
            # Unvisited state — no meaningful arrow to draw
            ax.text(x + 0.5, plot_y + 0.5, '·',
                    ha='center', va='center', fontsize=8, color='lightgray')
        else:
            dx, dy = arrow_map[greedy_actions[state]]
            ax.annotate("", 
                        xy=(x + 0.5 + dx, plot_y + 0.5 + dy),
                        xytext=(x + 0.5 - dx, plot_y + 0.5 - dy),
                        arrowprops=dict(arrowstyle="->", color='steelblue', lw=1.2))

    # ── Step 3: simulate and draw the greedy path from each starting position ─
    # Starting states from reset(): (x=2, y=2) and (x=2, y=9)
    starting_positions = [(2, 2), (2, 9)]
    path_colors = ['blue', 'red']
    path_labels = ['Start 1 (top, y=2)', 'Start 2 (bottom, y=9)']

    for (start_x, start_y), color, label in zip(starting_positions, path_colors, path_labels):
        xs_path, ys_path = [], []
        x, y = start_x, start_y
        visited = set()

        # Walk greedily until goal or cycle detected (safety valve)
        for _ in range(r * c * 2):
            state = y * c + x
            if state in visited:
                break  # cycle — stop to avoid infinite loop
            visited.add(state)

            # Record center of current cell (flipped y for plotting)
            xs_path.append(x + 0.5)
            ys_path.append(r - 1 - y + 0.5)

            if env.s[y, x] == 'G':
                break  # reached goal

            # Apply greedy action using the same logic as env.step()
            action = greedy_actions[state]
            if action == 0 and y > 0:       y -= 1  # up
            elif action == 1 and y < r - 1: y += 1  # down
            elif action == 2 and x > 0:     x -= 1  # left
            elif action == 3 and x < c - 1: x += 1  # right

            # If agent hits cliff, it teleports back to start
            if env.s[y, x] == 'C':
                break  # show path up to cliff and stop

        ax.plot(xs_path, ys_path, color=color, linewidth=6.5,
                marker='o', markersize=4, label=label, alpha=0.2, zorder=5)
        # Mark the starting cell with a star
        ax.plot(xs_path[0], ys_path[0], color=color,
                marker='*', markersize=15, zorder=6)

    #  Step 4: draw grid lines ──────────────────────────────────────────────
    for i in range(c + 1):
        ax.axvline(i, color='gray', linewidth=0.4, zorder=1)
    for i in range(r + 1):
        ax.axhline(i, color='gray', linewidth=0.4, zorder=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150)  # save for report
    plt.show()


def smooth(returns, window=20):
    # np.ones(window)/window creates an averaging kernel of size 'window'
    # 'valid' mode means output only contains values where the window fits fully
    return np.convolve(returns, np.ones(window)/window, mode='valid')

if __name__ == "__main__":
    # --- Single long run to visualize the greedy policy ---
    env = [ShortcutEnvironment(), WindyShortcutEnvironment()]

    agent1 = QLearningAgent(
        n_actions=env[1].action_size(),
        n_states=env[1].state_size(),
        env=env[1]
    )
    # agent1.train(n_episodes=10000)
    # env[0].render_greedy(agent1.Q)  # visualize what the agent learned
    # plot_greedy_policy_with_paths(agent1.Q, env[0], title="Q-Learning Greedy Policy")

    agent2 = SARSAAgent(
        n_actions=env[1].action_size(),
        n_states=env[1].state_size(),
        env=env[1]
    )
    # agent2.train(n_episodes=10000)
    # env[0].render_greedy(agent2.Q)  # visualize what the agent learned
    # plot_greedy_policy_with_paths(agent2.Q, env[0], title="SARSA Greedy Policy")

    # learning curve over 100 repetitions of 1000 episodes
    # mean_returns = run_repetitions(env=env[0], agent_type="QLearning", n_rep=100, n_episodes=1000, alpha=0.1)

    # plt.plot(mean_returns, label="alpha=0.1")
    # plt.xlabel("Episode")
    # plt.ylabel("Cumulative Reward")
    # plt.title("Q-Learning: Average Return over 100 Repetitions")
    # plt.show()

    # comparison hyperparameters 
    # plt.figure()
    # alphas = [0.01, 0.1, 0.5, 0.9]
    # episodes = [100, 500, 1000, 10000]
    # epsilons = [0.01, 0.1, 0.5, 0.9]
    # for a in alphas:
    #     mean_returns = run_repetitions(env=env[0], agent_type="QLearning", n_rep=100, n_episodes=1000, alpha=a)
    #     plt.plot(smooth(mean_returns), label=f"alpha={a}")
    # plt.xlabel("Episode")
    # plt.ylabel("Cumulative Reward")
    # plt.title("Q-Learning: Average Return over different alphas")
    # plt.legend()
    # plt.show()

    # SARSA comparison
    # plt.figure()
    # alphas = [0.01, 0.1, 0.5, 0.9]
    # agents = ["QLearning", "SARSA"]
    # for a in agents:
    #     mean_returns = run_repetitions(env=env[0], agent_type=a, n_rep=100, n_episodes=1000, alpha=0.1)
    #     plt.plot(mean_returns, label=f"alpha={a}")
    # plt.xlabel("Episode")
    # plt.ylabel("Cumulative Reward")
    # plt.title("SARSA: Average Return over 100 Repetitions")
    # plt.legend()
    # plt.show()

    # stormy weather
    agents = ["QLearning", "SARSA"]
    for a in agents:
        if a == "QLearning":
            agent1.train(n_episodes=10000)
            plot_greedy_policy_with_paths(agent1.Q, env[1], title="Q-Learning Greedy Policy, Stormy Weather")
        elif a == "SARSA":
            agent2.train(n_episodes=10000)
            plot_greedy_policy_with_paths(agent2.Q, env[1], title="SARSA Greedy Policy, Stormy Weather")
    
    