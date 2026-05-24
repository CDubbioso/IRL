#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent, PrioritizedSweepingAgent
from Helper import LearningCurvePlot, smooth


def run_repetitions(method, n_repetitions, n_timesteps, eval_interval,
                    learning_rate, gamma, epsilon, n_planning_updates,
                    wind_proportion, max_episode_length=100,
                    n_eval_episodes=30):
    """ Run n_repetitions independent runs of the chosen agent ('dyna' or 'ps').
    Returns:
        eval_timesteps: array of timesteps at which evaluation was performed.
        mean_returns:   the average evaluation return at each of those timesteps,
                        averaged over the repetitions.
        mean_runtime:   the mean wall-clock time per repetition (seconds).
    """
    eval_timesteps = np.arange(0, n_timesteps, eval_interval)
    n_evals = len(eval_timesteps)
    eval_returns = np.zeros((n_repetitions, n_evals))
    runtimes = np.zeros(n_repetitions)

    for rep in range(n_repetitions):
        # Each repetition: brand new environment and agent 
        env = WindyGridworld(wind_proportion=wind_proportion)
        eval_env = WindyGridworld(wind_proportion=wind_proportion)

        if method == 'dyna':
            agent = DynaAgent(env.n_states, env.n_actions, learning_rate, gamma)
        elif method == 'ps':
            agent = PrioritizedSweepingAgent(env.n_states, env.n_actions,
                                             learning_rate, gamma)
        else:
            raise ValueError(f"Unknown method: {method}")

        s = env.reset()
        eval_idx = 0

        t0 = time.time()
        for t in range(n_timesteps):
            # Evaluate at every eval_interval timesteps 
            if t % eval_interval == 0:
                eval_returns[rep, eval_idx] = agent.evaluate(
                    eval_env,
                    n_eval_episodes=n_eval_episodes,
                    max_episode_length=max_episode_length,
                )
                eval_idx += 1

            a = agent.select_action(s, epsilon)
            s_next, r, done = env.step(a)
            agent.update(s=s, a=a, r=r, done=done,
                         s_next=s_next,
                         n_planning_updates=n_planning_updates)

            if done:
                s = env.reset()
            else:
                s = s_next

        runtimes[rep] = time.time() - t0

    mean_returns = eval_returns.mean(axis=0)
    return eval_timesteps, mean_returns, runtimes.mean()

def planning_budget_experiment(method, wind_proportion, n_planning_updatess,
                               n_repetitions, n_timesteps, eval_interval,
                               learning_rate, gamma, epsilon,
                               max_episode_length, smoothing_window,
                               title, filename):
    """ Learning-curve experiment for one method on one environment,
    sweeping over different planning budgets 

    Returns a dict mapping n_planning_updates -> (eval_timesteps, mean_returns, mean_runtime).
    """
    print(f"\n=== {title} ===")
    plot = LearningCurvePlot(title=title)

    results = {}
    for k in n_planning_updatess:
        if k == 0:
            label = 'Q-learning (K=0)'
            run_method = 'dyna'
        else:
            label = f'{method.upper()} (K={k})'
            run_method = method

        print(f"  running {label} ...")
        eval_t, mean_ret, runtime = run_repetitions(
            method=run_method,
            n_repetitions=n_repetitions,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            n_planning_updates=k,
            wind_proportion=wind_proportion,
            max_episode_length=max_episode_length,
        )
        print(f"    mean runtime / rep: {runtime:.2f} s, "
              f"final eval mean: {mean_ret[-1]:.2f}")
        smoothed = smooth(mean_ret, smoothing_window)
        plot.add_curve(eval_t, smoothed, label=label)
        results[k] = (eval_t, mean_ret, runtime)

    plot.save(name=filename)
    print(f"  saved {filename}")
    return results


def best_planning_key(results, exclude=(0,)):
    """ Pick the best planning budget from a results dict by final-window
    mean evaluation return """
    best_k, best_score = None, -np.inf
    for k, (eval_t, mean_ret, _) in results.items():
        if k in exclude:
            continue
        # Use the mean over the final 20% of evaluations
        tail = max(1, int(0.2 * len(mean_ret)))
        score = np.mean(mean_ret[-tail:])
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def comparison_plot(dyna_results, ps_results, label_env, filename,
                    smoothing_window):
    """ Build a comparison plot for one environment with the best Dyna,
    best PS, and Q-learning baseline. """
    plot = LearningCurvePlot(title=f'Comparison ({label_env})')

    eval_t, mean_ret, _ = dyna_results[0]
    plot.add_curve(eval_t, smooth(mean_ret, smoothing_window),
                   label='Q-learning (K=0)')

    best_dyna_k = best_planning_key(dyna_results, exclude=(0,))
    eval_t, mean_ret, _ = dyna_results[best_dyna_k]
    plot.add_curve(eval_t, smooth(mean_ret, smoothing_window),
                   label=f'Dyna (K={best_dyna_k})')

    best_ps_k = best_planning_key(ps_results, exclude=(0,))
    eval_t, mean_ret, _ = ps_results[best_ps_k]
    plot.add_curve(eval_t, smooth(mean_ret, smoothing_window),
                   label=f'Prioritized Sweeping (K={best_ps_k})')

    plot.save(name=filename)
    print(f"  saved {filename}")
    return best_dyna_k, best_ps_k


def write_runtime_table(dyna_results_stoch, ps_results_stoch,
                        dyna_results_det, ps_results_det,
                        best_dyna_stoch, best_ps_stoch,
                        best_dyna_det, best_ps_det,
                        filename):
    """ Write a small text table comparing runtimes per repetition. """
    lines = []
    lines.append("Average runtime per repetition (single learning instance)")
    lines.append("=" * 72)
    lines.append(f"{'Algorithm':<35}{'Stochastic (s)':>18}{'Deterministic (s)':>18}")
    lines.append("-" * 72)
    qlearn_stoch = dyna_results_stoch[0][2]
    qlearn_det = dyna_results_det[0][2]
    lines.append(f"{'Q-learning (n=0)':<35}{qlearn_stoch:>18.2f}{qlearn_det:>18.2f}")

    dyna_stoch = dyna_results_stoch[best_dyna_stoch][2]
    dyna_det = dyna_results_det[best_dyna_det][2]
    lines.append(f"{'Dyna (best K stoch=' + str(best_dyna_stoch) + ', det=' + str(best_dyna_det) + ')':<35}{dyna_stoch:>18.2f}{dyna_det:>18.2f}")

    ps_stoch = ps_results_stoch[best_ps_stoch][2]
    ps_det = ps_results_det[best_ps_det][2]
    lines.append(f"{'Prioritized Sweeping (best K stoch=' + str(best_ps_stoch) + ', det=' + str(best_ps_det) + ')':<35}{ps_stoch:>18.2f}{ps_det:>18.2f}")

    text = "\n".join(lines)
    with open(filename, "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"  saved {filename}")


def experiment():
    # Hyperparameters 
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 20
    gamma = 1.0
    learning_rate = 0.2
    epsilon = 0.1
    max_episode_length = 100
    smoothing_window = 5  # over 41 evaluation points

    wind_proportions = [0.9, 1.0]
    n_planning_updatess = [1, 3, 5]
    
    # We add 0 to recover Q-learning as the model-free baseline
    sweep = [0] + n_planning_updatess

    # Part 1: Dyna planning-budget experiment 
    dyna_results_stoch = planning_budget_experiment(
        method='dyna', wind_proportion=0.9,
        n_planning_updatess=sweep,
        n_repetitions=n_repetitions, n_timesteps=n_timesteps,
        eval_interval=eval_interval, learning_rate=learning_rate,
        gamma=gamma, epsilon=epsilon,
        max_episode_length=max_episode_length,
        smoothing_window=smoothing_window,
        title='Dyna - stochastic environment (wind_proportion=0.9)',
        filename='dyna_stochastic.png',
    )
    dyna_results_det = planning_budget_experiment(
        method='dyna', wind_proportion=1.0,
        n_planning_updatess=sweep,
        n_repetitions=n_repetitions, n_timesteps=n_timesteps,
        eval_interval=eval_interval, learning_rate=learning_rate,
        gamma=gamma, epsilon=epsilon,
        max_episode_length=max_episode_length,
        smoothing_window=smoothing_window,
        title='Dyna - deterministic environment (wind_proportion=1.0)',
        filename='dyna_deterministic.png',
    )

    # Part 2: Prioritized sweeping planning-budget experiment
    ps_results_stoch = planning_budget_experiment(
        method='ps', wind_proportion=0.9,
        n_planning_updatess=sweep,
        n_repetitions=n_repetitions, n_timesteps=n_timesteps,
        eval_interval=eval_interval, learning_rate=learning_rate,
        gamma=gamma, epsilon=epsilon,
        max_episode_length=max_episode_length,
        smoothing_window=smoothing_window,
        title='Prioritized Sweeping - stochastic environment (wind_proportion=0.9)',
        filename='ps_stochastic.png',
    )
    ps_results_det = planning_budget_experiment(
        method='ps', wind_proportion=1.0,
        n_planning_updatess=sweep,
        n_repetitions=n_repetitions, n_timesteps=n_timesteps,
        eval_interval=eval_interval, learning_rate=learning_rate,
        gamma=gamma, epsilon=epsilon,
        max_episode_length=max_episode_length,
        smoothing_window=smoothing_window,
        title='Prioritized Sweeping - deterministic environment (wind_proportion=1.0)',
        filename='ps_deterministic.png',
    )

    # Part 3: Comparison plots 
    best_dyna_stoch, best_ps_stoch = comparison_plot(
        dyna_results_stoch, ps_results_stoch,
        label_env='wind_proportion=0.9',
        filename='comparison_stochastic.png',
        smoothing_window=smoothing_window,
    )
    best_dyna_det, best_ps_det = comparison_plot(
        dyna_results_det, ps_results_det,
        label_env='wind_proportion=1.0',
        filename='comparison_deterministic.png',
        smoothing_window=smoothing_window,
    )

    # Runtime table 
    write_runtime_table(
        dyna_results_stoch, ps_results_stoch,
        dyna_results_det, ps_results_det,
        best_dyna_stoch, best_ps_stoch,
        best_dyna_det, best_ps_det,
        filename='runtime_table.txt',
    )


if __name__ == '__main__':
    experiment()
