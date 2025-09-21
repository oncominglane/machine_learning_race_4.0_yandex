# coding: utf-8

import numpy as np

n_states = 500 # for Taxi-v3
n_actions = 6 # for Taxi-v3


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    elite_states = []
    elite_actions = []
    threshold = np.percentile(rewards_batch, percentile)
    #print(threshold)

    for i in range(len(rewards_batch)):
        if rewards_batch[i] >= threshold:
            elite_states.extend(states_batch[i])
            elite_actions.extend(actions_batch[i])

    assert elite_states is not None and elite_actions is not None
    return elite_states, elite_actions


def update_policy(elite_states, elite_actions, n_states=n_states, n_actions=n_actions):
    counts = np.zeros((n_states, n_actions), dtype=np.float64)
    # counts[s, a] = сколько раз в элитных данных в состоянии s выбирали действие a

    for i in range(len(elite_states)):
        s = elite_states[i]
        a = elite_actions[i]
        counts[s,a] += 1.0

    new_policy = np.zeros((n_states, n_actions), dtype=np.float64) 

    for s in range(n_states):
        row_sum = counts[s].sum()
        if row_sum > 0:
            new_policy[s] = counts[s] / row_sum
        else:
            new_policy[s] = np.ones(n_actions) / n_actions

    assert new_policy is not None
    return new_policy


def generate_session(env, policy, t_max=int(10**4)):

    states, actions = [], []
    total_reward = 0.

    s, info = env.reset()

    for t in range(t_max):
        # your code here - sample action from policy and get new state, reward, done flag etc. from the environment
        a = np.random.choice([i for i in range(policy.shape[1])], p=policy[s])
        new_s, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        assert new_s is not None and r is not None and done is not None
        assert a is not None
        # your code here
        # Record state, action and add up reward to states,actions and total_reward accordingly.
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done:
            break
    return states, actions, total_reward