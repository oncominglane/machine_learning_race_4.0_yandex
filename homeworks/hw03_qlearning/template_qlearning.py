import numpy as np
import random
from collections import defaultdict


def my_softmax(values: np.ndarray, T=1.):
    """
    Compute softmax with temperature for numerical stability.
    
    Args:
        values: np.array of shape (n,) - input values
        T: float - temperature parameter (default 1.0)
    
    Returns:
        np.array of shape (n,) - softmax probabilities                                                        V
    """
    # your code here
    mmax = np.max(values)
    values_new = values - mmax #for numerical stability
    values_new = values_new / T
    
    exp_values = np.exp(values_new)
    sum_exp_values = np.sum(exp_values)

    probas =  exp_values / sum_exp_values

    assert probas is not None
    return probas


class QLearningAgent:
    def __init__(self, alpha, discount, get_legal_actions, temp=1.):
        """
        Q-Learning Agent
        based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Instance variables you have access to
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)

        Functions you should use
          - self.get_legal_actions(state) {state, hashable -> list of actions, each is hashable}
            which returns legal actions for a state
          - self.get_qvalue(state,action)Q
            which returns Q(state,action)
          - self.set_qvalue(state,action,value)
            which sets Q(state,action) := value
        !!!Important!!!
        Note: please avoid using self._qValues directly.
            There's a special self.get_qvalue/set_qvalue for that.
        """

        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.discount = discount
        self.temp = temp

    def get_qvalue(self, state, action):
        """Returns Q(state,action)"""
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """Sets the Qvalue for [state,action] to the given value"""
        self._qvalues[state][action] = value

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        # Calculate the approximation of value function V(s).
        Q_values = []
        for a in possible_actions:
            Q_values.append(self.get_qvalue(state=state, action=a))

        value = max(Q_values)
        assert value is not None
        return value


    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha
        
        # Calculate the updated value of Q(s, a).
        qtarget = reward + gamma * self.get_value(next_state)
        
        qvalue = self.get_qvalue(state, action)
        qvalue = (1-learning_rate) * qvalue + learning_rate * qtarget
        assert qvalue is not None

        self.set_qvalue(state, action, qvalue)

    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # Choose the best action wrt the qvalues.
        qvalue = np.array([self.get_qvalue(state, a) for a in possible_actions], dtype=float)
        qvalue_max = np.max(qvalue)

        for a in possible_actions:
            if self.get_qvalue(state, a) == qvalue_max:
                best_action = a
                assert best_action is not None
                return best_action

        

    def get_softmax_policy(self, state):
        """
        Compute all actions probabilities in the current state according
        to their q-values using softmax policy.

        Actions probability should be computed as
        p(a_i|s) = softmax([q(s, a_1), q(s, a_2), ... q(s, a_k)])_i
        Softmax temperature is set to `self.temp`.
        See the formula in the notebook for more details
        """
        possible_actions = self.get_legal_actions(state)
        # If there are no legal actions, return None
        llen = len(possible_actions)
        if llen == 0:
            return None
        
        # Compute all actions probabilities in the current state using softmax
        q_values = np.empty(llen, dtype=float)
        for i in range(llen):
            q_values[i] = self.get_qvalue(state=state, action=possible_actions[i])
        assert q_values is not None

        probabilities = my_softmax(values=q_values, T=self.temp)
        assert probabilities is not None

        return probabilities


    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        Select actions according to softmax policy.

        Note: To pick randomly from a list, use np.random.choice(..., p=actions_probabilities)
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """
        possible_actions = self.get_legal_actions(state)
        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # Select the action to take in the current state according to the policy
        actions_probabilities = self.get_softmax_policy(state)
        chosen_action = np.random.choice(possible_actions, p=actions_probabilities)
        assert chosen_action is not None
        return chosen_action


class EVSarsaAgent(QLearningAgent):
    """
    An agent that changes some of q-learning functions to implement Expected Value SARSA.
    Note: this demo assumes that your implementation of QLearningAgent.update uses get_value(next_state).
    If it doesn't, please add
        def update(self, state, action, reward, next_state):
            and implement it for Expected Value SARSA's V(s')
    """

    def get_value(self, state):
        """
        Returns Vpi for current state under the softmax policy:
          V_{pi}(s) = sum _{over a_i} {pi(a_i | s) * Q(s, a_i)}

        Hint: all other methods from QLearningAgent are still accessible.
        """
        possible_actions = self.get_legal_actions(state)
        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        
        # Compute the value of the current state under the softmax policy.
        Q_values = []
        for a in possible_actions:
            Q_values.append(self.get_qvalue(state=state, action=a))

        actions_probabilities = self.get_softmax_policy(state)

        value = np.dot(actions_probabilities, Q_values)
        assert value is not None

        return value
