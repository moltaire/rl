import numpy as np


class AgentVars:
    # This class specifies the agent parameters

    def __init__(self):
        """ This function defines the instance variable unique to each instance
        """
        self.alpha_win = 0.7
        self.alpha_loss = 0.5
        self.beta = 1  # The paper does not mention an inverse temperature parameter, but it should be included


class Agent:
    def __init__(self, agent_vars):
        self.agent_vars = agent_vars
        self.options = [0, 1]
        self.v_a_t = np.zeros(len(self.options))
        self.a_t = None

    def __repr__(self):
        return f"Dual learning rate agent with\n  alpha_win = {self.agent_vars.alpha_win}\n  alpha_loss = {self.agent_vars.alpha_loss}\n  beta = {self.agent_vars.beta}"

    def softmax(self, v_a_t):
        """This function implements the softmax choice rule.

        Args:
            v_a_t (array): Current action values

        Returns:
            p_a_t: Choice probabilities
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
        if r_t == 1:
            alpha = self.agent_vars.alpha_win
        elif r_t == 0:
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
    agent_vars = AgentVars()
    agent = Agent(agent_vars=agent_vars)
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

