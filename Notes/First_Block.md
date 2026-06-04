# Reinforcement Learning
Computational approach to learn from interaction 
- involves an agent that learns through trial-and-error and delayed rewards, making decision to maximise long term rewards. 

**KEY ASPECT**: **Eploration - Exploitatio** tradeoff

RL $\rightarrow$ State - Action - Reward

**KEY ELEMENTS**:
- Policy $\rightarrow$ strategy to select an action 
- Reward Function $\rightarrow$ signals immediate outcome (good, bad)
- Value Function $\rightarrow$ measures long-term expected rewards 

**BANDIT** : simplified RL problem where an agent repeatedly chooses between several actions, each giving a random reward drawn from an unknown distribution. 
- Goal $\rightarrow$ maximise total reward over time by finding out which action is best $\rightarrow$ balance exploration and eploitation 

--- 
## Bandits
$\rightarrow$ ***one-step decision making problem*** 

Bandit Environment : each action leads to a single reward, and th enevironment doesn't evolve in a new state
- every action is independent $\rightarrow$ don't influence future rewards

Standard Environment : actions change and influence future rewards

$\Rightarrow$ both share exploration-exploitation tradeoff

---

### BANDIT
$$\langle A, p(r,a) \rangle$$
- $A$ set of discrete actions
- $p(r,a)$ conditional probability distribution that maps each action to a distribution over possible rewards

---

### $Q(a)$ - Action Value **or** Expected Reward **of an action** 

$$Q(a) = \mathbb{E}_{r \sim p(r,a)} [r]$$

- average reward you'd expect from choosing action $a$ if you pulled it infinitely many times
- is the true value of that arm, but since you don't know "$p(r,a)$", you have to estimate it $\rightarrow$ _experience_ 

Initialization of $Q(a)$
- **Realistic Initialization** $\rightarrow$ initial guess that the reward for each action is $0$

$$Q(a) = 0 \qquad \forall a \in A$$

- **Optimistic Initialization** $\rightarrow$ encurages exploration by starting with a high value for each action 
    - algorithm is more likely to explore less tried action, since they appear to have high expected rewards initially

$$Q(a) = \phi \qquad \forall a \in A$$

---

### $\pi(a)$ - Policy **or** Action Selection

$$\pi(a) \in [0,1] \qquad \forall a \in A \qquad , \qquad \sum_{a \in A} \pi(a) = 1$$

- a policy $\pi(a)$ is a probability distribution over the action space $A$, where each action "$a$" is assigned a probability of being selected

- defines the agent's compete strategy $\rightarrow$ $\pi(a)$ tells exactly how likely the agent is too choose each available action 
    - different policies just define different ways od distribution that probability across the actions

* ***Explicit policy*** : directly stores the probabilities in $\pi(a)$
* ***Implicit policy*** : probabilities are derived from Q-values

---

### Estimate a Mean

$$Q(n) = \frac{1}{n} \sum_{i=1}^n r_i$$

$\rightarrow$ mean of the first $n$ rewards

How to update the formula for $n+1$ ?
- **Incremental Update** :

$$Q_n = Q_{n-1} + \frac{1}{n} [r_n - Q_{n-1}]$$

- **Learning Update** : 
    - $\alpha$ $\rightarrow$ makes recent rewards carry more weight that older rewards, because they get multiplied by $\alpha$ more often
        - $\alpha$ large : mean jumps to the new reward
        - $\alpha$ small : new reward don't have large impact

$$Q(n) \leftarrow Q_{n-1} + \alpha [r_n - Q_{n-1}]$$

---

### Bandit Algorithms
3 things we have to decide:
1. **initial estimates** of $Q(a)$ $\rightarrow$ realistic or optimistic
2. **policy** $\rightarrow$ how to select actions (exploration-exploitation)
3. **update rule** 

### 1. $\epsilon$-Greedy policy 

$$ \pi_{\epsilon-\text{greedy}}(a) = \begin{cases} 1 - \epsilon, & \text{if } a = \arg\max_{b \in A} Q(b) \\ 
\frac{\epsilon}{|A|-1}, & \text{otherwise} \end{cases} $$

- balances exploration and exploitation by chosing the best action with probability $1-\epsilon$, and exploring other actions randomly with probability $\epsilon$ 
- $\epsilon \in [0,1]$, scales the amount od exploration 

### 2. **Greedy Policy with** Optimistic Initialization - OI 

$$ \pi_{\text{greedy}}(a) = \begin{cases} 1, & \text{if } a = \arg\max_{b \in A} Q(b) \\ 
0, & \text{otherwise} \end{cases} $$

- this selects the best known action with certanty 
- OI ensures that initially all actions are explored by setting high initial estimates for $Q(a)$

### 3. UCB - Upper Confidence Bound

$$ \pi_{\text{UCB}}(a) = \begin{cases} 1, & \text{if } a = \arg\max_{a} \left[ Q(a) + c \cdot \sqrt{\frac{\ln t}{n(a)}} \right] \\ 
0, & \text{otherwise} \end{cases} $$

where:
- $c$ controls the amount of exploration $\rightarrow$ $c$ bigger = more exploration
- $t$ denotes the timesteps 
- $n(a)$ is the number of times action "$a$" has been taken 

$\rightarrow \pi_{UCB}$ returns an action rather than a probability of an action. 

---
## Markov Decision Process - MPD
$\rightarrow$ ***sequential decision-making problem***

MPD $\rightarrow$ actions influence future states and rewards, we may prefer low instant reward if it gives us high long term reward
- can handle stochastic environments $\rightarrow$ through probabilistic transition functions
- can tradeoff multiple goals $\rightarrow$ through reward function 

Markov Property $\rightarrow$ "_The future only depends on the present and not on past history_"

| **Item** | **Symbol** | **Description**  |
|------------------------|---------------|------------------------------|
| **State Space** | $S$ | All possible states the system can be in |
| **Action Space** | $A$ | All possible actions the agent can take |
| **Transition Function** |$p(s'\|s,a)$ | Probability of transitioning to a state given an action |
| **Reward Function** | $r(s,a,s')$ | Immediate reward for transitioning between states |
| **Discount Factor** | $γ$ | Factor that determines how much future rewards are valued |
- **Policy**: Describes how the agent chooses actions in different states ($\pi (a \| s)$).
- **Trace**: A sequence of state-action-reward pairs induced by the policy ($\tau$).
- **Return**: The cumulative sum of rewards from a trace ($R(\tau)$).
- **Value**: The expected return from each state or action under a policy ($v^{\pi}(s) , q^{\pi}(s,a)$).
- **Optimal Value/Policy**: There exists one optimal value function with a greedy policy that maximizes the return ($v^{*}(s) , q^{*}(s,a) , \pi^{*}(s)$).


### STATE
| ***Atomic*** | ***Factorized*** |
| ------ | ---------- |
| unique states |  |
| no relation between states| state is a set of related features |

For Atomic state and action space, the transition function can be sotred as an array of size : $|S| \times |A| \times |S|$ 

### POLICY 
$\rightarrow$ probability of taking an action in a given state

- **RANDOM** :each action has equal probability of being selected
- **DETERMISTIC** : always choose same action in given state
- **GREEDY** : choose action that maximise expected reward

### Trace
$\rightarrow$ sequence state-action-reward 

$$\tau = s_t , a_t , r_t , s_{t+1}, a_{t+1}, r_{t+1}, ... , s_{t+n}, a_{t+n}, r_{t+n} $$

***Trace Probabiliy*** : multilpy all individual policies and transition probabilities in the trace. 

$$ p(\tau) = \pi(a_t | s_t) \cdot p(r_t, s_{t+1} | s_t, a_t) \cdot \pi(a_{t+1} | s_{t+1}) \cdot p(r_{t+1}, s_{t+2} | s_{t+1}, a_{t+1}) \dots $$

### RETURN 
$\rightarrow$ sum of rewards of a trace
$\rightarrow$ down-weight long-term rewards $\rightarrow$ Discounted Return 

$$ R(\tau) = r_t + r_{t+1} + r_{t+2} \dots \qquad \rightarrow \qquad R(\tau) = r_t + \gamma \cdot r_{t+1} + {\gamma}^2 \cdot r_{t+2} \dots $$

$\gamma \to 1$ future rewards have higher weight

$\gamma \to 0$ future rewards have smaller weight

### Infinite Horizon 
$\rightarrow$ sum of rewards infinitely unless a terminal state is reached

$$R(\tau) = \sum_{i=0}^{\infty} \gamma^i \cdot r_{t+i}$$

### VALUE 
$\rightarrow$ expected return when starting from "$s$" with a policy "$\pi$" 

$$ v^{\pi}(s) = \mathbb{E}_{\tau \sim p^{\pi}(\tau)} [r_t + \gamma \cdot r_{t+1} + \gamma^2 \cdot r_{t+2} + \dots | s_t = s] $$ 

- this gives the expected total reward form a state, considering future rewards
- each policy has its own value function shown with $v^{\pi}(s)$



---
## Dynamic Programming
$\rightarrow$ you need complete knowledge of the environment

Algorithm that computes optima decisions in environments that can be built as Markov Decision Processes (MDPs) 
- DP acts like a bridge between planning and learning through experience
- DP takes full knowledge of the environment using transition probabilities $p(s' | s,a)$ and expected reward $r(s,a,s')$

How to compute the optimal policy $\pi^{*}$ ?
1. **Policy Iteration**
2. **Value Iteration**

### Policy Iteration 
1. Initialize a random policy.
2. **Policy Evaluation**: Calculate the value of each state for the current policy.
3. **Policy Improvement**: Update the policy by choosing the best action (greedy) for each state based on its value.
4. Repeat steps 2 and 3 until the policy converges (value at each state don't change anymore).

**Policy Evaluation** 
- Goal : given a fixed policy, compute its value function $v^{\pi}(s)$
- How?
    - iteratively apply Bellman Equation as an update rule

Bellman Equation for $v(s)$:

$$ v^{\pi}(s) = \mathbb{E}_{a \sim \pi(a|s)} \mathbb{E}_{s' \sim p(s'|a,s)} [r(s,a,s') + \gamma \cdot v^{\pi}(s')] $$

Bellman Equation for $q(s,a)$:

$$ q^{\pi}(s,a) = \mathbb{E}_{s' \sim p(s'|a,s)} [r(s,a,s') + \gamma \cdot \mathbb{E}_{a' \sim \pi(a'|s')} [q^{\pi}(s',a')]] $$

Bellman Equation from building blocks:

$$ v^{\pi}(s) = \mathbb{E}_{a \sim \pi(a|s)} [q^{\pi}(s,a)] \qquad q^{\pi}(s,a) = \mathbb{E}_{s' \sim p(s'|a,s)} [r(s,a,s') + \gamma v^{\pi}(s')] $$


### Policy Improvement 
Given a value function find a better policy 

How? $\rightarrow$ act greedily
- for $q(s,a)$

$$\pi^{\text{greedy}}(s) = \text{argmax}_a q^{\pi}(s,a)$$

- for $v(s)$ 

$$\pi^{\text{greedy}}(s) = \text{argmax}_a \mathbb{E}_{s' \sim p(s'|a,s)} [r(s,a,s') + \gamma \cdot v(s')]$$


### Value Iteration 
Combine _policy evaluation_ and _policy improvement_ into a single step. 
- Instead of separately evaluating and then improving the policy, value iteration iteratively updates the value function until it converges to the optimal values for all states. From these values, you can derive the optimal policy.


---
### **Key Differences**:
1. **Policy Iteration**:
   - Alternates between **policy evaluation** and **policy improvement**.
   - Converges faster in practice, but requires **two steps per iteration** (policy evaluation and improvement).
   - **Policy Iteration** tends to converge **faster** than value iteration when the state space is small to medium.

2. **Value Iteration**:
   - Performs a **single step** that combines both evaluation and improvement.
   - Can take longer because it updates the value function iteratively until it converges, but it doesn't require explicit policy evaluation.
   - **Value Iteration** can be more efficient when the problem requires fewer iterations, but it may take longer to converge on larger state spaces.