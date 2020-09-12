import numpy as np


class TaskVars:
    # This class specifies the task parameters

    def __init__(self):
        """This function defines the instance variables
           following Kahnt, Park et al. (2008)
        """

        # Set up general task properties
        self.n_trials = 100  # Number of trials per session
        self.n_sessions = 2  # Number of session
        self.n_targets = 2  # Number of targets (options) per trial

        # Set up task rules (states)
        # Each state contains probabilities of reward for each action (p_r_a)
        # and the a list of correct responses.
        self.task_rules = {
            "0": {"p_r_0": 0.2, "p_r_1": 0.8, "a_correct": [1]},
            "1": {"p_r_0": 0.8, "p_r_1": 0.2, "a_correct": [0]},
        }

        # Set up reversal properties
        self.n_trials_reversal_min = 10  # Min. number of trials before reversal
        self.p_correct_reversal_min = 0.7  # Min. p_correct needed before reversal
        self.n_trials_reversal_max = 16  # Max. number of trials before reversal


class Task:
    # This class represents the task

    def __init__(self, task_vars):
        """Initialize a task with a set of task variables.

        Args:
            task_vars (TaskVars): Set of task variables
        """
        self.task_vars = task_vars
        # By default, initialize a task with a random rule (state)
        self.rule_t = np.random.choice(list(task_vars.task_rules.keys()))
        self.n_trials_current_rule = 0  # Count trials with current task rule
        self.n_correct_current_rule = 0  # and number of correct responses
        self.p_correct_current_rule = 0
        self.correct_t = None

    def __repr__(self):
        return f"Task with the task rules:\n  {self.task_vars.task_rules}"


    def get_p_r_a(self, a_t):
        """This function returns the probability of reward, given the agents' action and the current task rule.

        Args:
            a_t (str): The agent's current action
        """
        p_r_a = self.task_vars.task_rules[self.rule_t][f"p_r_{a_t}"]
        return p_r_a

    def sample_reward(self, a_t):
        """This function returns a probabilistic reward, given the agents' action and the current task rule.

        Args:
            a_t (str): The agent's current action
        """
        p_r = self.get_p_r_a(a_t)
        self.r_t = np.random.binomial(1, p_r)

        # Update reversal variables
        self.n_trials_current_rule += 1
        self.correct_t = a_t in self.task_vars.task_rules[self.rule_t]["a_correct"]
        self.n_correct_current_rule += int(self.correct_t)
        self.p_correct_current_rule = (
            self.n_correct_current_rule / self.n_trials_current_rule
        )

    def check_reversal(self, verbose=False):
        """This functions checks if a reversal must occur, according to the task variables and the agents' choice history (number of trials with current rule and level of p_correct)
        """
        # Reversal when minimum number of trials was reached with minimum accuracy
        if (self.n_trials_current_rule > self.task_vars.n_trials_reversal_min) & (
            self.p_correct_current_rule >= self.task_vars.p_correct_reversal_min
        ):
            reversal_t = True
        # Reversal if maximum number of trials reached without minimum accuracy
        elif self.n_trials_current_rule == self.task_vars.n_trials_reversal_max:
            reversal_t = True
        else:
            reversal_t = False

        # If a reversal occurs, choose a new rule at random (don't choose the current rule again)
        # Implement other possible mechanisms that govern rule reversal here.
        if reversal_t:
            current_rule = self.rule_t
            other_rules = [
                rule
                for rule in list(self.task_vars.task_rules.keys())
                if rule != current_rule
            ]
            new_rule = np.random.choice(other_rules)
            if verbose:
                print(f"\n/!\ Reversal: {current_rule} -> {new_rule}\n")
            self.rule_t = new_rule
            self.n_trials_current_rule = 0  # Reset trials with current task rule
            self.n_correct_current_rule = 0  # and number of correct responses
            self.p_correct_current_rule = 0


if __name__ == "__main__":

    np.random.seed(1)

    # Task setup
    task_vars = TaskVars()
    task_vars.n_trials = 50
    task_vars.n_blocks = 1
    task = Task(task_vars=task_vars)

    for trial in range(task.task_vars.n_trials):
        print(f"Trial {trial}")
        print(f"  Rule: {task.rule_t}")

        # Make a (random) choice
        a_t = np.random.choice([0, 1])
        print(f"  Action: {a_t}")

        # Return reward
        task.sample_reward(a_t)
        print(f"  Reward: {task.r_t}")

        # Print statistics
        print(f"  Trials in current rule: {task.n_trials_current_rule}")
        print(f"  P(correct) in current rule: {task.p_correct_current_rule:.2f}")

        # Check for reversals
        task.check_reversal(verbose=True)
