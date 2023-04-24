from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Optional, Hashable

from rl2023.constants import EX1_CONSTANTS as CONSTANTS
from rl2023.exercise1.mdp import MDP, Transition, State, Action


class MDPSolver(ABC):
    """Base class for MDP solvers

    **DO NOT CHANGE THIS CLASS**

    :attr mdp (MDP): MDP to solve
    :attr gamma (float): discount factor gamma to use
    :attr action_dim (int): number of actions in the MDP
    :attr state_dim (int): number of states in the MDP
    """

    def __init__(self, mdp: MDP, gamma: float):
        """Constructor of MDPSolver

        Initialises some variables from the MDP, namely the state and action dimension variables

        :param mdp (MDP): MDP to solve
        :param gamma (float): discount factor (gamma)
        """
        self.mdp: MDP = mdp
        self.gamma: float = gamma

        self.action_dim: int = len(self.mdp.actions)
        self.state_dim: int = len(self.mdp.states)

    def decode_policy(self, policy: Dict[int, np.ndarray]) -> Dict[State, Action]:
        """Generates greedy, deterministic policy dict

        Given a stochastic policy from state indeces to distribution over actions, the greedy,
        deterministic policy is generated choosing the action with highest probability

        :param policy (Dict[int, np.ndarray of float with dim (num of actions)]):
            stochastic policy assigning a distribution over actions to each state index
        :return (Dict[State, Action]): greedy, deterministic policy from states to actions
        """
        new_p = {}
        for state, state_idx in self.mdp._state_dict.items():
            new_p[state] = self.mdp.actions[np.argmax(policy[state_idx])]
        return new_p

    @abstractmethod
    def solve(self):
        """Solves the given MDP
        """
        ...


class ValueIteration(MDPSolver):
    """MDP solver using the Value Iteration algorithm
    """

    def _calc_value_func(self, theta: float) -> np.ndarray:
        """Calculates the value function

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        **DO NOT ALTER THE MDP HERE**

        Useful Variables:
        1. `self.mpd` -- Gives access to the MDP.
        2. `self.mdp.R` -- 3D NumPy array with the rewards for each transition.
            E.g. the reward of transition [3] -2-> [4] (going from state 3 to state 4 with action
            2) can be accessed with `self.R[3, 2, 4]`
        3. `self.mdp.P` -- 3D NumPy array with transition probabilities.
            *REMEMBER*: the sum of (STATE, ACTION, :) should be 1.0 (all actions lead somewhere)
            E.g. the transition probability of transition [3] -2-> [4] (going from state 3 to
            state 4 with action 2) can be accessed with `self.P[3, 2, 4]`

        :param theta (float): theta is the stop threshold for value iteration
        :return (np.ndarray of float with dim (num of states)):
            1D NumPy array with the values of each state.
            E.g. V[3] returns the computed value for state 3
        """
        V = np.zeros(self.state_dim)
        ### PUT YOUR CODE HERE ###
        update_values = True
        while update_values:
            change_in_state_values = 0
            for state_index, state in enumerate(self.mdp.states):
                old_state_value = V[state_index]
                action_quality = {}
                for action_index, action in enumerate(self.mdp.actions):
                    q_s_a = 0
                    possible_next_states_from_s_under_a = self.mdp.P[state_index, action_index, :]
                    for next_state_index,prob_of_next_state in enumerate(possible_next_states_from_s_under_a):
                        reward = self.mdp.R[state_index, action_index, next_state_index]
                        q_s_a += prob_of_next_state*(reward + CONSTANTS['gamma']*V[next_state_index])

                    action_quality[action_index] = q_s_a

                best_action = max(action_quality, key=action_quality.get)
                V[state_index] = action_quality[best_action]
                change_in_state_values = max(abs(old_state_value - V[state_index]), change_in_state_values)
            if change_in_state_values < theta:
                update_values = False


        return V

    def _calc_policy(self, V: np.ndarray) -> np.ndarray:
        """Calculates the policy

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        :param V (np.ndarray of float with dim (num of states)):
            A 1D NumPy array that encodes the computed value function (from _calc_value_func(...))
            It is indexed as (State) where V[State] is the value of state 'State'
        :return (np.ndarray of float with dim (num of states, num of actions):
            A 2D NumPy array that encodes the calculated policy.
            It is indexed as (STATE, ACTION) where policy[STATE, ACTION] has the probability of
            taking action 'ACTION' in state 'STATE'.
            REMEMBER: the sum of policy[STATE, :] should always be 1.0
            For deterministic policies the following holds for each state S:
            policy[S, BEST_ACTION] = 1.0
            policy[S, OTHER_ACTIONS] = 0
        """
        policy = np.zeros([self.state_dim, self.action_dim])
        ### PUT YOUR CODE HERE ###
        for state_index, state in enumerate(self.mdp.states):
            action_quality = {}
            for action_index, action in enumerate(self.mdp.actions):
                q_s_a = 0
                possible_next_states_from_s_under_a = self.mdp.P[state_index, action_index, :]
                for next_state_index, prob_of_next_state in enumerate(possible_next_states_from_s_under_a):
                    reward = self.mdp.R[state_index, action_index, next_state_index]
                    q_s_a += prob_of_next_state * (reward + CONSTANTS['gamma'] * V[next_state_index])

                action_quality[action_index] = q_s_a

            best_action = max(action_quality, key=action_quality.get)
            policy[state_index][best_action] = 1

        return policy

    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        Compiles the MDP and then calls the calc_value_func and
        calc_policy functions to return the best policy and the
        computed value function

        **DO NOT CHANGE THIS FUNCTION**

        :param theta (float, optional): stop threshold, defaults to 1e-6
        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)):
            Tuple of calculated policy and value function
        """
        self.mdp.ensure_compiled()
        V = self._calc_value_func(theta)
        policy = self._calc_policy(V)

        return policy, V


class PolicyIteration(MDPSolver):
    """MDP solver using the Policy Iteration algorithm
    """

    def _policy_eval(self, policy: np.ndarray) -> np.ndarray:
        """Computes one policy evaluation step

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        :param policy (np.ndarray of float with dim (num of states, num of actions)):
            A 2D NumPy array that encodes the policy.
            It is indexed as (STATE, ACTION) where policy[STATE, ACTION] has the probability of
            taking action 'ACTION' in state 'STATE'.
            REMEMBER: the sum of policy[STATE, :] should always be 1.0
            For deterministic policies the following holds for each state S:
            policy[S, BEST_ACTION] = 1.0
            policy[S, OTHER_ACTIONS] = 0
        :return (np.ndarray of float with dim (num of states)): 
            A 1D NumPy array that encodes the computed value function
            It is indexed as (State) where V[State] is the value of state 'State'
        """
        V = np.zeros(self.state_dim)
        ### PUT YOUR CODE HERE ###
        repeat = True
        while repeat:
            change_in_state_values = 0
            for state_index, state in enumerate(self.mdp.states):
                new_state_value = 0
                old_state_value = V[state_index]
                for action_index, action in enumerate(self.mdp.actions):
                    pi_s_given_a = policy[state_index][action_index]
                    q_s_a = 0
                    possible_next_states_from_s_under_a = self.mdp.P[state_index, action_index, :]
                    for next_state_index, prob_of_next_state in enumerate(possible_next_states_from_s_under_a):
                        reward = self.mdp.R[state_index, action_index, next_state_index]
                        q_s_a += prob_of_next_state * (reward + CONSTANTS['gamma'] * V[next_state_index])
                    new_state_value += pi_s_given_a * q_s_a
                V[state_index] = new_state_value
                change_in_state_values = max(change_in_state_values, abs(new_state_value-old_state_value))
            if change_in_state_values < self.theta:
                repeat = False

        return np.array(V)

    def _policy_improvement(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes policy iteration until a stable policy is reached

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        Useful Variables (As with Value Iteration):
        1. `self.mpd` -- Gives access to the MDP.
        2. `self.mdp.R` -- 3D NumPy array with the rewards for each transition.
            E.g. the reward of transition [3] -2-> [4] (going from state 3 to state 4 with action
            2) can be accessed with `self.R[3, 2, 4]`
        3. `self.mdp.P` -- 3D NumPy array with transition probabilities.
            *REMEMBER*: the sum of (STATE, ACTION, :) should be 1.0 (all actions lead somewhere)
            E.g. the transition probability of transition [3] -2-> [4] (going from state 3 to
            state 4 with action 2) can be accessed with `self.P[3, 2, 4]`

        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)):
            Tuple of calculated policy and value function
        """
        policy = np.zeros([self.state_dim, self.action_dim])
        V = np.zeros([self.state_dim])
        ### PUT YOUR CODE HERE ###

        policy_not_stable = True
        if self.action_dim != 0:
            policy = np.ones([self.state_dim, self.action_dim]) * np.divide(1, self.action_dim)

        while policy_not_stable:
            policy_not_stable = False
            V = self._policy_eval(policy)
            for state_index, state in enumerate(self.mdp.states):
                policy_for_state = policy[state_index]
                if np.all(policy_for_state == policy_for_state[0]):
                    old_action = np.random.randint(self.action_dim)
                else:
                    old_action = np.argmax(policy[state_index])

                new_action_qualities = {}
                for action_index, action in enumerate(self.mdp.actions):
                    action_quality = 0
                    possible_next_states_from_s_under_a = self.mdp.P[state_index, action_index, :]
                    for next_state_index, prob_of_next_state in enumerate(possible_next_states_from_s_under_a):
                        reward = self.mdp.R[state_index, action_index, next_state_index]
                        action_quality += prob_of_next_state * (reward + CONSTANTS['gamma'] * V[next_state_index])
                    new_action_qualities[action_index] = action_quality

                best_action = max(new_action_qualities, key=new_action_qualities.get)

                for action_index, action in enumerate(self.mdp.actions):
                    policy[state_index][action_index] = 0
                policy[state_index][best_action] = 1

                if best_action != old_action:
                    policy_not_stable = True

        return policy, V

    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        This function compiles the MDP and then calls the
        policy improvement function that the student must implement
        and returns the solution

        **DO NOT CHANGE THIS FUNCTION**

        :param theta (float, optional): stop threshold, defaults to 1e-6
        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)]):
            Tuple of calculated policy and value function
        """
        self.mdp.ensure_compiled()
        self.theta = theta
        return self._policy_improvement()


if __name__ == "__main__":
    mdp = MDP()
    mdp.add_transition(
        #         start action end prob reward
        Transition("rock0", "jump0", "rock0", 1, 0),
        Transition("rock0", "stay", "rock0", 1, 0),
        Transition("rock0", "jump1", "rock0", 0.1, 0),
        Transition("rock0", "jump1", "rock1", 0.9, 0),
        Transition("rock1", "jump0", "rock1", 0.1, 0),
        Transition("rock1", "jump0", "rock0", 0.9, 0),
        Transition("rock1", "jump1", "rock1", 0.1, 0),
        Transition("rock1", "jump1", "land", 0.9, 10),
        Transition("rock1", "stay", "rock1", 1, 0),
        Transition("land", "stay", "land", 1, 0),
        Transition("land", "jump0", "land", 1, 0),
        Transition("land", "jump1", "land", 1, 0),
    )

    solver = ValueIteration(mdp, CONSTANTS["gamma"])
    policy, valuefunc = solver.solve()
    print("---Value Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function")
    print(valuefunc)

    solver = PolicyIteration(mdp, CONSTANTS["gamma"])
    policy, valuefunc = solver.solve()
    print("---Policy Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function")
    print(valuefunc)