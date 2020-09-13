import numpy as np
from scipy.optimize import minimize

from rl.agent import Agent, AgentVars


class EstimationVars:
    def __init__(self, task_vars):
        """ This function defines the instance variable unique to each instance
                      n_trials: Number of trials
                      n_sim: Number of simulations
                      n_sp: Number of starting points
                      alpha_win_bnds: Estimation boundaries for alpha_win parameter
                      alpha_win_fixedsp: Fixed starting point for alpha_win parameter
                      alpha_loss_bnds: Estimation boundaries for alpha_loss parameter
                      alpha_loss_fixedsp: Fixed starting point for alpha_loss parameter
                      beta_bnds: Estimation boundaries for beta parameter
                      beta_fixedsp: Fixed starting point for beta parameter
                      rand_sp: Indicate if you want to use random starting points
        """
        self.n_trials = task_vars.n_trials
        self.n_blocks = task_vars.n_blocks
        self.n_sim = np.nan
        self.n_sp = 1
        self.n_params = 3
        self.alpha_win_bnds = (0, 1)
        self.alpha_loss_bnds = (0, 1)
        self.beta_bnds = (0, 20)
        self.alpha_win_fixedsp = 0.5
        self.alpha_loss_fixedsp = 0.5
        self.beta_fixedsp = 5
        self.rand_sp = True


class Estimation:
    def __init__(self, est_vars):
        self.est_vars = est_vars

    def llh(self, x, data, agent_vars):
        """This function computes the log-likelihood of choices

        Args:
            x (np.array): Free parameters
            data (pd.DataFrame): DataFrame containing choice (`a`) and reward (`r`) data.
            agent_vars (rl.agent.AgentVars): Agent specific variables.
        """

        # Name parameter inputs
        alpha_win, alpha_loss, beta = x

        # Initialize likelihood array
        llh_a = np.full([self.est_vars.n_trials, self.est_vars.n_blocks], np.nan)

        # Cycle over blocks
        for b in range(self.est_vars.n_blocks):

            # Extract required data
            a = data[data["block"] == b]["a"].values
            r = data[data["block"] == b]["r"].values

            # Agent initialization
            agent = Agent(agent_vars)

            # Assign current parameters
            agent.agent_vars.alpha_win = alpha_win
            agent.agent_vars.alpha_loss = alpha_loss
            agent.agent_vars.beta = beta

            # Initialize block specific variables
            cp = np.full(self.est_vars.n_trials, np.nan)  # choice probability

            # Cycle over trials
            for t in range(0, self.est_vars.n_trials):

                # Evaluate probability of economic decisions
                agent.decide()

                # Set decision
                agent.a_t = np.int(a[t])

                # Extract probability of current decision
                cp[t] = agent.p_a_t[np.int(a[t])]

                # Compute log likelihood of economic decision
                llh_a[t, b] = np.log(agent.p_a_t[np.int(a[t])])

                # Agent contingency parameter update
                agent.learn(np.int(r[t]))

        # Sum negative log likelihoods
        llh = -1 * np.sum(llh_a)

        return llh

    def estimate(self, data, seed=None):

        # Set RNG seed
        np.random.seed(seed)

        agent_vars = AgentVars()

        # Initialize values
        min_llh = np.inf  # "Best" initial LLH, overwritten by first estimation run
        min_x = np.nan  # And no parameter estimate

        # Cycle over starting points
        for r in range(self.est_vars.n_sp):

            # Estimate parameters
            bounds = [
                self.est_vars.alpha_win_bnds,
                self.est_vars.alpha_loss_bnds,
                self.est_vars.beta_bnds,
            ]

            # Set starting points
            if self.est_vars.rand_sp:
                x0 = [np.random.uniform(b[0], b[1]) for b in bounds]
            else:
                x0 = [
                    self.est_vars.alpha_win_fixedsp,
                    self.est_vars.alpha_loss_fixedsp,
                    self.est_vars.beta_fixedsp,
                ]

            # Run the estimation
            result = minimize(self.llh, x0, args=(data, agent_vars), method="L-BFGS-B", bounds=bounds)

            # Extract maximum likelihood parameter estimate
            x = result.x

            # Extract minimized negative log likelihood
            llh = result.fun

            # Check if cumulated negative log likelihood is lower than the previous
            # one and select the lowest
            if llh < min_llh:
                min_llh = llh
                min_x = x

        # Compute BIC for economic decision
        bic = self.compute_bic(min_llh, self.est_vars.n_params)

        return min_llh, bic, min_x.tolist()

    def compute_bic(self, llh, n_params):
        """ This function compute the Bayesian information criterion (BIC)
            See Stephan et al. (2009). Bayesian model selection for group studies. NeuroImage
        :param llh: Negative log likelihood
        :param n_params: Number of free parameters
        :return: bic
        """

        bic = (-1 * llh) - (n_params / 2) * np.log(
            self.est_vars.n_trials * self.est_vars.n_blocks
        )

        return bic
