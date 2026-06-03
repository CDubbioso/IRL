# Model-Based RL

## Access to MPD Dynamics
The dynamic function $p(s'|s,a)$ says which next state $s'$ can follow when you take an action $a$ in state $s$. The kind of access you are allowed to this function decides which alorithm you can run.

**Access** is **Reversible or Irreversible**?
- **Reversible access** = a **model** $\rightarrow$ you can try an action in any state and rewind. 
- **Irreversible access** = an **environment** $\rightarrow$ once you take an action, you must continue from the next state. 

Do you get the **full distribution** or a **single sample**? 
- **Distribution model** : you get the full probability distribution over the next state
- **Sample model** : you get only one sample of the next state

$\rightarrow$ an environment always give a sample 

---
## Planning vs Learning 
The distinction between planning and learning lies in two factors:
1. **Access to MPD Dynamics** $\rightarrow$ **reversible** or **irreversible**
2. **Storage of the solution** $\rightarrow$ **local** or **global**
    - **local** - stores a solution for a subset of all states
        - focus on current state
        - discarded after execution
        - ex: planning tree
    - **global** - solution permanently stores estimates for all states 
        - ex: value table

Problem : we have two possible distinctions

|    | **Local** solution | **Global** solution |
|--------------|-------------------|---------------|
| **Reversible MDP access**   | Planning (e.g., MCTS) | Borderline/Model-based RL (e.g., Dynamic Programming) |
| **Irreversible MDP access** | (impossible) | Model-free RL (e.g., Q-learning) |

Is impossible to use a local solution when we have irreversible access because local solutions get discarded after execution of the real action, but it the environment is irreversible, we directly throw away our new solution after the first sample. 

---
## Back-Ups
In RL, a ***back-up*** refers to how we update the value function or action-value function when estimating the value of a state or action

- **Expected back-up** $\rightarrow$ used in planning where you have a model and can compute the full expected value over the next state
- **Sample Back-up** $\rightarrow$ used in RL when you only have access to sampled experiences 

### Back-Up Diagrams
The back-up diagrams visually show how updates are made in the state values $V(s)$ and state-action values $Q(s,a)$. The diagrams correspond to different types of updates.

- On-Policy vs Off-Policy:
    - On-Policy : the value or Q-values are updated using the same policy that is being followed (e.g., Sarsa where the policy is followed directly)
    - Off-policy: the value or Q-values are updated using a different policy than the one being followed (e.g., Q-learning where the greedy policy is used to update, but the agent may be exploring randomly)

### 1-Step Back-Up $\rightarrow$ Shallow Update
- These updates involve only a single step (transition from one state to the next) and are shallow in terms of the depth of the update
- TD(0) and Monte Carlo methods are examples of shallow updates because they only consider the immediate next state or the final return (in the case of Monte Carlo)

### Multi-Step Back-Up - Deep Update
- Multi-step updates involve looking further into the future, considering multiple steps in the environment (depth)
- ex : Monte Carlo learning looks at complete episodes, while TD(λ) or dynamic programming methods may use longer sequences of states and actions to perform updates
- These updates are deeper and often lead to more accurate estimates of value functions but are computationally more expensive

--- 

**Notes:**
* **Expected back-ups** are used in **planning** where you have a model and can compute the full expected value over the next state.
* **Sample back-ups** are used in **RL** when you only have access to sampled experiences.
* **Shallow updates** look at immediate outcomes (1-step), while **deep updates** consider longer-term effects (multi-step).

---

## Model-Based Reinforcement Learning

1. **How to learn a model form data?**
2. **How to integrate planning updates and learning updates?**

### 1. Learning a Model
- Given a dataset of observed transitions $(s,a,r,s')$, how can we estimate:
    1. the **dynamics** $p(s'|s,a)$ and 
    2. the **reward** $r(s,a,s')$ **function** ?

--- 

**Dynamic Estimation** 
- track counts $n(s,a,s')$ 
    - number of times we observed $s'$ after taking $a$ in $s$ 
    - can estimate from transition data $(s,a,r,s')$
- estimate $p(s'|s,a)$ by normalizing the observed counts:

$$ \hat{p}(s'|s, a) = \frac{n(s, a, s')}{n(s, a)} $$ 

**Reward Function Estimation** 
- track total transition rewards $R_{\text{sum}}(s,a,s')$ 
    - Sum of all observed rewards when reaching $s'$ after taking $a$ in $s$
    - can estimate from transition data $(s,a,r,s')$ 
- Estimate $r(s,a,s')$ be computing the average transition reward:

$$ \hat{r}(s, a, s') = \frac{R_{\text{sum}}(s, a, s')}{n(s, a, s')} $$

$\rightarrow$ store two arrays, $n(s,a,s')$ and $R_{\text{sum}}(s,a,s')$, each of size $|S| \cdot |A| \cdot |S|$ 

--- 

## Model-Based RL Algorithms
$\rightarrow$ combine **planning** and **learning** by using a learned model to simulate the environment. 

### 1. Real-Time Dynamic Programming - RTDP
Simple Dynamic Progrmming:
- sweep th whole state space
- update each state with Bellman Equation untill convergence
- Main Problem : ***Curse of Dimentionality*** 
    - $2^{nd}$ problem : many states are often not even reachable from start

- RTDP updates values using the Bellman optimality equation, similar to Dynamic Programming, but focuses on reachable states instead of all states in the space.
- **Curse of Dimensionality**: Traditional DP is inefficient in large state spaces because it updates all states, even unreachable ones. RTDP solves this by focusing only on states that are actually visited
- **Efficient Updates**: RTDP applies updates only to states that are part of the trajectory from the start, making it more efficient and scalable
- **Uniform Updates**: While traditional DP updates all states uniformly, RTDP targets only reachable states, improving performance in large problems

$\rightarrow$ ***RTDP reduces unnecessary calculations*** by focusing on reachable states, making it more efficient for large state spaces.

---
### 2. Dyna 
Idea $\rightarrow$ "learn a model, use it to generate extra simulated transitions, and apply standard update to both real and simulated transitions"

1. **learn a model** of the environment
2. use the model to simulate **one-step planning updates** to our value function 

The algorithm updates the Q-values using both actual environment samples and simulated samples from the learned model. This helps improve data efficiency by leveraging the model.

Algorithm details for Dyna Q-learning:

1. Initialize Q-values, transition counts, and reward sums.
2. For each timestep:
   * Take an action using an $\epsilon$-greedy policy.
   * Observe the transition and update the model.
   * Perform Q-learning updates using real experiences.
   * Perform planning updates using simulated transitions from the model.

---
### 3. Prioritized Sweeping
If the value estimate of a state changes a lot, then the states that precede it should probably also be updated. 

Idea: 
- "use a backward/reverse model to identify states that likely need updating"
- "prioritize states that deserve an update & additional backward search"

Prioritized sweeping focuses on efficiently updating the value function by identifying **states with high-priority updates**:
* When the Q-value estimate for a state-action pair changes significantly, the predecessor states that lead to this state should also be updated.
* This prioritization helps focus updates on the most promising state-action pairs.

**Mechanism**:
- keep a **priority queue** with priority equal to the absolute TD error
- inster a state-action when its priority passes a threshold $\theta$
- pop the highest $\rightarrow$ update it $\rightarrow$ use the reverse model to find predecessors and queue the ones taht also pass $\theta$
$\rightarrow$ new information than races backwards along the paths that lead to it. 

$$ p \leftarrow | r + \gamma \cdot \max_{a'}\hat{Q}(s',a') - \hat{Q}(s,a) | $$

where:
- $r + \gamma \cdot \max_{a'}\hat{Q}(s',a')$ is the ***new Q-learning estimate*** 
- $\hat{Q}(s,a)$ is the ***current estimate***

