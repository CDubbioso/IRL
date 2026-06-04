# Model-Free RL
Traditional MPDs assume we know everything about the environment. But we want to learn good policies without knowing everything in advance.

$\rightarrow$ Learning from experinece only 

Can we still find a good policy if we only have experience to learn from?
- yes, through:
    - **Monte Carlo** 
    - **Temporal Difference** - **TD**

## Monte Carlo methods
Goal: Estimate the value of states under a policy $\pi$

- Generating episodes
- Average observed return for each episode
    - you can then average the return for each specific state

--- 

### First-Visit MC 
- **Initialize state values** $v^{\pi}(s)$ by our desire.
- **Generate episodes**: You start at an initial state and take actions according to the policy which forms an episode (a sequence of states, actions, and rewards).
- **For each state visited in the episode**, update its value using the **cumulative discounted rewards** from that state onwards.
    - **The update rule**: For each state visited, calculate the average reward after its first occurrence.

$$ v^{\pi}(s) = \frac{(v^{\pi}(s) \text{ from previous episodes} + \text{new reward})}{\text{number of episodes}} $$

__Goal__: Refine state value estimates after multiple episodes.

Monte Carlo for Action Values:
- Use similar MC methods but calculate averages for **state-action pairs**.

**Exploration Issue**: some state-action pairs may not be visited
- Solution: exploring starts from a random state-action pair

**First-Visit vs Every-Visit MC**
- **First-Visit MC**: Updates the value of a state only after its first occurrence in an episode.
- **Every-Visit MC**: Updates the value of a state every time it is visited in an episode.
- Example:
    - **First-Visit**: If a state is visited multiple times in an episode, only the first visit's reward is used for the update.
    - **Every-Visit**: All visits to the state in the episode contribute to the update.

--- 

### Generalised Policy Iteration
$\rightarrow$ to learn a good policy (what actions to take)

GPI Process:
- **Policy Evaluation**: Use MC methods to estimate the value of the current policy.
- **Policy Improvement**: Update the policy by acting greedily with respect to the estimated values.
- Repeat until convergence.


---
## On-Policy vs Off-Policy
**On-policy**: The policy used to generate episodes is the same as the one being optimized.
- **Advantages**: Simple and easy to implement.
- **Disadvantages**: Can be suboptimal due to constant exploration.

**Off-policy**: The policy used to generate episodes differs from the one being optimized.
- **Advantages**: More powerful and flexible.
- **Disadvantages**: More complicated and slower to converge.
    - In off-policy learning, the behavior policy $b$ (the policy used to generate episodes) differs from the target policy $\pi$ (the policy being optimized).
    - **Problem**: If an action is not taken in the behavior policy, the value for that state-action pair is unknown.
    - **Solution**: **Importance sampling** adjusts the returns to account for differences in probabilities between $b$ and $\pi$

|           | **On-Policy**                      | **Off-Policy**                        |
| --------- | ---------------------------------- | ------------------------------------- |
| Acts with | the pollicy being improved         | behaviour policy $b$                  |
| Improves  | the same policy                    | a separate target policy $\pi$        |
| Pros      | simpler, stable                    | general; target can be greedy         |
| Cons      | stays suboptimal (keeps exploring) | more complex, higher variance, slower | 


---
## Temporal Difference
MC methods wait for the end of the episode to start learning
- can be a problem for long episodes, or environmnets without episodes

Solution : **Temporal Difference Learning** - **TD**

- **Goal**: Avoid waiting until the end of an episode to update values. 
    - TD learning allows for updating values incrementally during the episode.
- **TD($0$)**: A one-step update method where the value of a state is updated based on the next step:

A simple MC Method update looks like:

$$v_{\pi}(s_t) \leftarrow v_{\pi}(s_t) + \alpha \left[G_t - v_{\pi}(s_t) \right]$$

$\rightarrow$ but $G_t$ is only known after the episode ends 

Solution : **TD(0)**
- one-step target built from the immediate reward plus the discounted estimate of the next state:

$$v_{\pi}(s_t) \leftarrow v_{\pi}(s_t) + \alpha \left[ R_{t+1} + \gamma \cdot v_{\pi}(s_{t+1}) - v_{\pi}(s_t) \right]$$

---
## SARSA - on-policy TD control 
Sarsa is an on-policy method where the agent learns the value of the current policy:

$$ Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R_{t+1} + \gamma \cdot Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right] $$

- **Key Point**: Sarsa updates the Q-value based on the action taken in the next state, making it more conservative than Q-learning.

**Sarsa vs Q-learning**: Sarsa uses the current policy to update the Q-values, while Q-learning uses the maximum future Q-value, independent of the current policy.

---
## Q-Learning - off-policy TD control
Q-learning is an off-policy method where the agent learns the optimal policy using a greedy target:

$$ Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R_{t+1} + \gamma \cdot \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right] $$

- **Key Point**: The maximization bias occurs when the agent overestimates action values.
- **Solution**: **Double Q-learning** uses two independent Q-tables to reduce maximization bias.

---
## Expected SARSA
Replace the single sampled next action with its expectation under $\pi$. This removes the randomness coming from the next action, so it has lower variance than Sarsa. 

- An extension of SARSA where the agent uses the expected value of the next state-action pair, weighted by the policy's probabilities. This can be used both on-policy and off-policy.
- **Key Point**: Expected SARSA can provide a more stable learning process by averaging over possible actions rather than selecting the maximum.

$$ Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R_{t+1} + \gamma \sum_a (\pi(a|s_{t+1})Q(s_{t+1},a)) - Q(s_t, a_t) \right] $$

$\rightarrow$ It can be used on-policy or off-policy. If the target policy is **greedy**, Expected Sarsa becomes
exactly **Q-learning**.

---

| **Method**         | **Bootstraped next-state target**           | **Policy type** |
| ------------------ | ------------------------------------------- | --------------- |
| **SARSA**          | $Q(s',a')$ for the sampled next action $a'$ | on-policy       |
| **Expected SARSA** | $\sum_a \pi(a,s')A(s',a)$                   | either          |
| **Q-Learning**     | $\max_a Q(s',a)$                            | off-policy      |

--- 
## Maximization Bias and Double Q-learning
- **Maximization Bias**: Q-learning can overestimate action values due to always selecting the maximum Q-value in future states.
- **Double Q-learning**: Addresses this bias by maintaining two separate Q-tables and using one to select actions and the other to evaluate them, reducing overestimation.

---
## N-step TD 
- **N-step TD**: An extension of TD learning that uses multiple steps of temporal difference learning to update values instead of just one step, thus incorporating more information for better value estimation. 

n-step TD return: 

$$ G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma ^{n-1}R_{t+n} + \gamma ^n V_{t+n-1}(S_{t+n}) $$

n-step state-value update:

$$ V_{t+n} \leftarrow V_{t+n-1}(S_t) + \alpha \left[G_{t:t+n} - V_{t+n-1} (S_t) \right] $$ 

- **N-step SARSA**: Similar to N-step TD but applies to the SARSA algorithm, allowing for more accurate value updates by considering multiple steps of rewards.

$$ G_{t:t+n} = R_{t+1} + \dots + \gamma ^{n-1} R_{t+n} + \gamma ^{n}Q_{t+n-1} (S_{t+n}, A_{t+n}) $$
