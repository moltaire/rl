import numpy as np


class AgentVars:
    # This class specifies the agent parameters.

    def __init__(self, **kwargs):
        """Here, agent parameters are initialized.

        Example for an rl.agent.DualLearningRateAgent:
            AgentVars(alpha_pos=0.3, alpha_neg=0.2, beta=3)
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, **kwargs):
        """This method updates agent parameters.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)


class DualLearningRateAgent:
    """This class implements a Dual-Learning-Rate agent as described in Kahnt, Park et al. (2008) or Lefebvre et al. (2017).

    This agent needs the following agent_vars to be set:
        
        alpha_pos (float): Learning rate for gains.
        alpha_neg (float): Learning rate for losses.
        beta (float): Inverse temperature parameter of softmax choice rule.
            This parameter is not mentioned in Kahnt, Park et al. (2008), but used here anyway.
    """

    def __init__(self, agent_vars, n_options, n_states=1, variant="delta"):
        """Initialize the Dual-Learning-Rate agent.

        Args:
            agent_vars (rl.agent.AgentVars): Agent specific parameters. Must have `alpha_pos`, `alpha_neg` and `beta` attributes.
            n_options (int): Number of options in each state.
            n_states (int, optional): Number of states in the task. Defaults to 1.
            variant (str, one of `['delta', 'r']`, optional): Toggle between (`r`) the Kahnt, Park et al. (2008), where the learning rate differs between positive and negative rewards (r > 0 vs r <= 0) and (`delta`) the Lefebvre et al. (2017) variant, where the learning rate differs between positive and negative *prediction errors* (delta > 0 vs delta < 0). Defaults to the Lefebvre variant (`delta`).
        """
        self.check_agent_vars(agent_vars)
        self.agent_vars = agent_vars
        self.options = range(n_options)
        self.variant = variant
        # Initial values
        if not hasattr(agent_vars, "Q_init"):
            self.agent_vars.Q_init = np.zeros((n_states, n_options))
        self.Q_t = self.agent_vars.Q_init.copy()
        self.a_t = None  # Initial action
        self.s_t = 0  # Initial state

    def check_agent_vars(self, agent_vars):
        for var in ["alpha_pos", "alpha_neg", "beta"]:
            if not hasattr(agent_vars, var):
                raise ValueError(f"agent_vars is missing `{var}` attribute.")

    def __repr__(self):
        return f"Dual learning rate agent ({self.variant} variant) with\n  alpha_pos = {self.agent_vars.alpha_pos:.2f}\n  alpha_neg = {self.agent_vars.alpha_neg:.2f}\n  beta = {self.agent_vars.beta:.2f}"

    def softmax(self, Q_s_t):
        """This function implements the softmax choice rule.

        Args:
            Q_t (numpy.array): Current action values

        Returns:
            p_a_t (numpy.array): Choice probabilities
        """
        p_a_t = np.exp(Q_s_t * self.agent_vars.beta) / np.sum(
            np.exp(Q_s_t * self.agent_vars.beta)
        )
        return p_a_t

    def learn(self, r_t):
        """This function implements the agent's learning process.

        Args:
            r_t (int): Current reward
        """
        delta = r_t - self.Q_t[self.s_t, self.a_t]

        # Determine learning rate, depending on model variant...
        if self.variant == "r":
            reference_var = r_t
        elif self.variant == "delta":
            reference_var = delta

        # and value of the reference variable (r or delta)
        if reference_var > 0:
            alpha = self.agent_vars.alpha_pos
        elif reference_var <= 0:
            alpha = self.agent_vars.alpha_neg

        self.Q_t[self.s_t, self.a_t] += alpha * delta

    def decide(self):
        """This function implements the agent's choice.
        """
        self.p_a_t = self.softmax(self.Q_t[self.s_t, :])
        self.a_t = np.random.choice(self.options, p=self.p_a_t)

    def observe_state(self, task):
        self.s_t = task.show_state()


if __name__ == "__main__":

    np.random.seed(1)

    # "Task" setup
    n_trials = 100
    p_r = {0: 0.75, 1: 0.25}

    # Agent setup
    agent_vars = AgentVars(alpha_pos=0.1, alpha_neg=0.05, beta=3)
    agent = DualLearningRateAgent(agent_vars=agent_vars, n_options=2, variant="delta")
    print(agent)
    print(f"  Q_t: {agent.Q_t}")

    # Task performance
    for trial in range(n_trials):
        print(f"Trial {trial}")

        # Observe state
        # Usually: agent.observe_state(task), but we don't have a real task instsance here. Therefore we just set the state manually.
        agent.s_t = 0
        agent.decide()
        print(f"  a_t: {agent.a_t}")

        r_t = np.random.binomial(1, p=p_r[agent.a_t])
        print(f"  r_t: {r_t}")

        agent.learn(r_t)
        print(f"  s_t: {agent.s_t}")
        print(f"  Q_t: {agent.Q_t}")
        print(f"  p_a_t: {agent.p_a_t}")

