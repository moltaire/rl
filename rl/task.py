import numpy as np


class TaskVars:
    # This class specifies the task parameters

    def __init__(
        self, n_trials=None, n_blocks=None, n_options=None, n_states=1, **kwargs
    ):
        """This function initializes task variables.

        Args:
            n_trials (int, optional): Number of trials per block. Defaults to None.
            n_blocks (int, optional): Number of blocks. Defaults to None.
            n_options (int, optional): Number of options. Defaults to Nones.
            n_states (int, optional): Number of different states. Defaults to None.
            
            Additional task variables needed by a specific task (e.g., states, rewards) can be given as kwargs.
        """

        # Set up general task properties
        self.n_trials = n_trials
        self.n_blocks = n_blocks
        self.n_options = n_options
        self.n_states = n_states

        # Other parameters given as kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class MultipleStateTask:
    """This class represents an instrumental learning task, where options in different states (conditions) are associated with different reward probabilities.
    
    The class is modeled after the task used in Lefebvre et al. (2017) and Palminteri et al. (2015), where there are four task conditions. In each trial, participants choose between two symbols that are each associated with a reward (or loss) determined by the trial condition. 
    Conditions (and therefore symbols, reward probabilities and rewards) vary between trials.
    In sum, there are eight (4 conditions x 2 options) different symbols, for which participants have to learn the values by choosing and receiving outcomes.
    """

    def __init__(self, task_vars):
        """Initialize a multiple-state learning task, where different task states (conditions) with different choice options are defined and varied randomly between trials.

        Args:
            task_vars (rl.task.task_vars): Set of task variables.
        """
        self.kind = "multiple-state"
        self.check_task_vars(task_vars)
        self.task_vars = task_vars
        self.trial = 0  # trial counter
        self.block = 0  # block counter
        (
            self.task_vars.n_blocks,
            self.task_vars.n_trials,
        ) = self.task_vars.state_sequence.shape
        self.task_vars.n_states = len(task_vars.states)

    def check_task_vars(self, task_vars):
        # TODO: Implement this. Check task variables so that states and reversal parameters are defined.

        # Necessary:
        # states (dict)
        # state sequence (array): n_blocks x n_trials

        # Redundant:
        # n_trials is derived from state sequence
        # n_blocks is derived from state sequence

        return True

    def __repr__(self):
        return f"Multiple-state learning task with the states:\n  {self.task_vars.states}"

    def get_p_r_a(self, a_t):
        """This function returns the probability of reward, given the agents' action and the current task state.

        Args:
            a_t (int): The agent's current decision

        Returns:
            float: The probability of obtaining a reward.
        """
        p_r_a = self.task_vars.states[self.state_t]["p_r"][a_t]
        return p_r_a

    def sample_reward(self, a_t):
        """This function returns a probabilistic reward, given the agents' action and the current task state.

        Args:
            a_t (int): The agent's current action
        """
        p_r = self.get_p_r_a(a_t)
        self.r_t = np.random.choice(
            self.task_vars.states[self.state_t]["rewards"], p=[p_r, 1 - p_r]
        )

        # Update trial variables
        self.correct_t = a_t in self.task_vars.states[self.state_t]["a_correct"]

    def prepare_trial(self, verbose=False):
        """This function selects the next trial state (condition) and advances the trial and block counters.
        """
        self.state_t = self.task_vars.state_sequence[self.block, self.trial]

        # Update trial counter
        self.trial += 1

        # Check if end of block is reached
        if self.trial >= self.task_vars.n_trials:
            self.block += 1
            self.trial = 0

    def show_state(self):
        """This function shows the current state to an agent.
        In this task, the states are observable (different conditions have different symbols), therefore the true state is shown.

        Returns:
            int: Current state
        """
        return self.state_t


class ReversalLearningTask:
    """This class represents a reversal learning task, where different task states are defined and reversed based on a set of conditions.
    """

    def __init__(self, task_vars):
        """Initialize a task with a set of task variables.

        Args:
            task_vars (TaskVars): Set of task variables
        """
        self.kind = "reversal-learning"
        self.check_task_vars(task_vars)
        self.task_vars = task_vars

        # By default, initialize a task with a random state (rule)
        self.state_t = np.random.choice(list(task_vars.states.keys()))

        # Initialize variables to trigger reversals
        self.n_trials_current_state = 0  # Count trials in current task state
        self.n_correct_current_state = 0  # and number of correct responses
        self.p_correct_current_state = 0
        self.correct_t = None

    def check_task_vars(self, task_vars):
        # TODO: Implement this. Check task variables so that states and reversal parameters are defined.
        return True

    def __repr__(self):
        return f"Reversal learning task with the states (rules):\n  {self.task_vars.states}"

    def get_p_r_a(self, a_t):
        """This function returns the probability of reward, given the agents' action and the current task state.

        Args:
            a_t (int): The agent's current action
        
        Returns:
            float: The probability of obtaining a reward.
        """
        p_r_a = self.task_vars.states[self.state_t]["p_r"][a_t]
        return p_r_a

    def sample_reward(self, a_t):
        """This function returns a probabilistic reward, given the agents' action and the current task state.

        Args:
            a_t (int): The agent's current action
        """
        p_r = self.get_p_r_a(a_t)
        self.r_t = np.random.choice(
            self.task_vars.states[self.state_t]["rewards"], p=[p_r, 1 - p_r]
        )

        # Update reversal variables
        self.n_trials_current_state += 1
        self.correct_t = a_t in self.task_vars.states[self.state_t]["a_correct"]
        self.n_correct_current_state += int(self.correct_t)
        self.p_correct_current_state = (
            self.n_correct_current_state / self.n_trials_current_state
        )

    def prepare_trial(self, verbose=False):
        """This function 
        """
        # Reversal when minimum number of trials was reached with minimum accuracy
        if (self.n_trials_current_state > self.task_vars.n_trials_reversal_min) & (
            self.p_correct_current_state >= self.task_vars.p_correct_reversal_min
        ):
            reversal_t = True
        # Reversal if maximum number of trials reached without minimum accuracy
        elif self.n_trials_current_state == self.task_vars.n_trials_reversal_max:
            reversal_t = True
        else:
            reversal_t = False

        # If a reversal occurs, choose a new rule at random (don't choose the current rule again)
        # Implement other possible mechanisms that govern rule reversal here.
        if reversal_t:
            current_state = self.state_t
            other_states = [
                state
                for state in list(self.task_vars.states.keys())
                if state != current_state
            ]
            new_state = np.random.choice(other_states)
            if verbose:
                print(f"\n/!\ Reversal: State {current_state} -> State {new_state}\n")
            self.state_t = new_state
            self.n_trials_current_state = 0  # Reset trials with current task rule
            self.n_correct_current_state = 0  # and number of correct responses
            self.p_correct_current_state = 0

    def show_state(self):
        """This function shows the current state to an agent.
        In this task, the states are not observable: Every trial looks identical to the participant, independent of the true underlying state. Therefore the true state is not revealed, and this function always returns 0.

        Returns:
            int: 0, the same state, independent of the true current state state_t
        """
        return 0


if __name__ == "__main__":

    np.random.seed(1)

    # Task setup
    task_vars = TaskVars(n_trials=20, n_blocks=2, n_options=2)

    states = {
        "0": {
            "p_r": np.array([0.2, 0.8]),
            "a_correct": [1],
            "rewards": np.array([1, 0]),
        },
        "1": {
            "p_r": np.array([0.8, 0.2]),
            "a_correct": [0],
            "rewards": np.array([1, 0]),
        },
        "2": {
            "p_r": np.array([0.5, 0.5]),
            "a_correct": [0, 1],
            "rewards": np.array([1, 0]),
        },
    }

    task_vars.states = states
    task_vars.n_trials_reversal_min = 10
    task_vars.n_trials_reversal_max = 16
    task_vars.p_correct_reversal_min = 0.7

    task = ReversalLearningTask(task_vars=task_vars)
    print(task)

    for trial in range(task.task_vars.n_trials):

        # Prepare next trial: Check for reversals
        task.prepare_trial(verbose=True)

        print(f"Trial {trial}")
        print(f"  State: {task.state_t}")

        # Make a (random) choice
        a_t = np.random.choice(range(task.task_vars.n_options))
        print(f"  Action: {a_t}")

        # Return reward
        task.sample_reward(a_t)
        print(f"  Reward: {task.r_t}")

        # Print statistics
        print(f"  Trials in current state: {task.n_trials_current_state}")
        print(f"  P(correct) in current state: {task.p_correct_current_state:.2f}")

