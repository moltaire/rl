import numpy as np


class AgentVars:
    # This class specifies the agent parameters.

    def __init__(self, **kwargs):
        """Here, agent parameters are initialized.

        Example for an rl.agent.DualLearningRateAgent:
            AgentVars(alpha_win=0.3, alpha_loss=0.2, beta=3)
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, **kwargs):
        """This method updates agent parameters.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)


class DualLearningRateAgent:
    """This class implements a Dual-Learning-Rate agent as described in Kahnt, Park et al. (2008).

    This agent needs the following agent_vars to be set:
        
        alpha_win (float): Learning rate for gains.
        alpha_loss (float): Learning rate for losses.
        beta (float): Inverse temperature parameter of softmax choice rule.
            This parameter is not mentioned in Kahnt, Park et al. (2008), but used here anyway.
    """

    def __init__(self, agent_vars, n_options):
        """Initialize the Dual-Learning-Rate agent.

        Args:
            agent_vars (rl.agent.AgentVars): Agent specific parameters. Must have `alpha_win`, `alpha_loss` and `beta` attributes.
            n_options (int): Number of options to represent.
        """
        self.check_agent_vars(agent_vars)
        self.agent_vars = agent_vars
        self.options = range(n_options)
        self.v_a_t = np.zeros(n_options)  # Initial values
        self.a_t = None  # Initial action

    def check_agent_vars(self, agent_vars):
        for var in ["alpha_win", "alpha_loss", "beta"]:
            if not hasattr(agent_vars, var):
                raise ValueError(f"agent_vars is missing `{var}` attribute.")

    def __repr__(self):
        return f"Dual learning rate agent with\n  alpha_win = {self.agent_vars.alpha_win}\n  alpha_loss = {self.agent_vars.alpha_loss}\n  beta = {self.agent_vars.beta}"

    def softmax(self, v_a_t):
        """This function implements the softmax choice rule.

        Args:
            v_a_t (numpy.array): Current action values

        Returns:
            p_a_t (numpy.array): Choice probabilities
        """
        p_a_t = np.exp(self.agent_vars.beta * v_a_t) / np.sum(
            np.exp(v_a_t * self.agent_vars.beta)
        )
        return p_a_t

    def learn(self, r_t):
        """This function implements the agent's learning process.

        Args:
            r_t (int): Current reward
        """
        delta_a_t = r_t - self.v_a_t[self.a_t]
        if r_t > 0:
            alpha = self.agent_vars.alpha_win
        elif r_t <= 0:
            alpha = self.agent_vars.alpha_loss
        self.v_a_t[self.a_t] += alpha * delta_a_t

    def decide(self):
        """This function implements the agent's choice.
        """
        self.p_a_t = self.softmax(self.v_a_t)
        self.a_t = np.random.choice(self.options, p=self.p_a_t)


if __name__ == "__main__":

    np.random.seed(2)

    # "Task" setup
    n_trials = 100
    p_r = {0: 0.8, 1: 0.2}

    # Agent setup
    agent_vars = AgentVars(alpha_win=0.3, alpha_loss=0.1, beta=1)
    agent = DualLearningRateAgent(agent_vars=agent_vars, n_options=2)
    print(f"  v_a_t: {agent.v_a_t}")

    # Task performance
    for trial in range(n_trials):
        print(f"Trial {trial}")

        agent.decide()
        print(f"  a_t: {agent.a_t}")

        r_t = np.random.binomial(1, p=p_r[agent.a_t])
        print(f"  r_t: {r_t}")

        agent.learn(r_t)
        print(f"  v_a_t: {agent.v_a_t}")
        print(f"  p_a_t: {agent.p_a_t}")

