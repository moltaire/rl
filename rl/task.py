import numpy as np


class TaskVars:
    # This class specifies the task parameters

    def __init__(self, n_trials, n_blocks=1, n_options=2, **kwargs):
        """This function initializes task variables.

        Args:
            n_trials (int): Number of trials per block.
            n_blocks (int, optional): Number of blocks. Defaults to 1.
            n_options (int, optional): Number of options. Defaults to 2.
            
            Additional task variables needed by a specific task (e.g., states, rewards) can be given as kwargs.
        """

        # Set up general task properties
        self.n_trials = n_trials
        self.n_blocks = n_blocks
        self.n_options = n_options

        # Other parameters given as kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class ReversalLearningTask:
    # This class represents a reversal learning task, where different task_rules are defined and reversed based on a set of conditions.

    def __init__(self, task_vars):
        """Initialize a task with a set of task variables.

        Args:
            task_vars (TaskVars): Set of task variables
        """
        self.kind = "reversal-learning"
        self.check_task_vars(task_vars)
        self.task_vars = task_vars

        # By default, initialize a task with a random rule (state)
        self.state_t = np.random.choice(list(task_vars.states.keys()))

        # Initialize variables to trigger reversals
        self.n_trials_current_state = 0  # Count trials in current task state
        self.n_correct_current_state = 0  # and number of correct responses
        self.p_correct_current_state = 0
        self.correct_t = None

    def check_task_vars(self, task_vars):
        # Todo: Check task variables so that states and reversal parameters are defined.
        return True

    def __repr__(self):
        return f"Reversal learning task with the states (rules):\n  {self.task_vars.states}"

    def get_p_r_a(self, a_t):
        """This function returns the probability of reward, given the agents' action and the current task state.

        Args:
            a_t (str): The agent's current action
        """
        p_r_a = self.task_vars.states[self.state_t]["p_r"][a_t]
        return p_r_a

    def sample_reward(self, a_t):
        """This function returns a probabilistic reward, given the agents' action and the current task state.

        Args:
            a_t (str): The agent's current action
        """
        p_r = self.get_p_r_a(a_t)
        self.r_t = np.random.choice(
            [self.task_vars.reward, self.task_vars.noreward], p=[p_r, 1 - p_r]
        )

        # Update reversal variables
        self.n_trials_current_state += 1
        self.correct_t = a_t in self.task_vars.states[self.state_t]["a_correct"]
        self.n_correct_current_state += int(self.correct_t)
        self.p_correct_current_state = (
            self.n_correct_current_state / self.n_trials_current_state
        )

    def prepare_next_trial(self, verbose=False):
        """This functions checks if a reversal must occur, according to the task variables and the agents' choice history (number of trials with current rule and level of p_correct)
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


if __name__ == "__main__":

    np.random.seed(1)

    # Task setup
    task_vars = rl.task.TaskVars()
    task_vars.n_trials = 100
    task_vars.n_blocks = 2

    states = {
        "0": {"p_r": np.array([0.2, 0.8]), "a_correct": [1]},
        "1": {"p_r": np.array([0.8, 0.2]), "a_correct": [0]},
        "2": {"p_r": np.array([0.5, 0.5]), "a_correct": [0, 1]},
    }

    task_vars.states = states
    task_vars.n_trials_reversal_min = 10
    task_vars.n_trials_reversal_max = 16
    task_vars.p_correct_reversal_min = 0.7
    task_vars.reward = 1
    task_vars.noreward = 0

    task = ReversalLearningTask(task_vars=task_vars)

    for trial in range(task.task_vars.n_trials):
        print(f"Trial {trial}")
        print(f"  State: {task.state_t}")

        # Make a (random) choice
        a_t = np.random.choice([0, 1])
        print(f"  Action: {a_t}")

        # Return reward
        task.sample_reward(a_t)
        print(f"  Reward: {task.r_t}")

        # Print statistics
        print(f"  Trials in current state: {task.n_trials_current_state}")
        print(f"  P(correct) in current state: {task.p_correct_current_state:.2f}")

        # Check for reversals
        task.prepare_next_trial(verbose=True)
