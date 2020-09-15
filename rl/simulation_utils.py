import numpy as np
from scipy.stats import pearsonr


def simulate_normal_correlated_data(
    N,
    r=0.5,
    mu_a=0,
    mu_b=0,
    sig_a=1,
    sig_b=1,
    lim_a=None,
    lim_b=None,
    tolerance=0.05,
    seed=None,
    verbose=False,
):
    """This function simulates data from two normally distributed variables A and B with given correlation.
    
    Args:
        N (int): Number of data points
        r (float): Desired Pearson correlation
        mu_a, mu_b (float, optional): Mean values of A and B
        sig_a, sig_b (float, optional): Standard deviations of A and B
        lim_a, lim_b (tuple): Limits of A and B
        tolerance (float, optional): Tolerated difference between desired and generated correlation
        seed (int, optional): Random seed
        verbose (bool, optional): Toggle verbosity
    
    Returns:
        numpy.array: N by 2 array with simulated values
    """
    np.random.seed(seed)

    # Construct mean vector
    mu = np.array([mu_a, mu_b])

    # Construct covariance matrix
    sig = np.array([[sig_a ** 2, r * sig_a * sig_b], [r * sig_a * sig_b, sig_b ** 2]])

    # Simulate data
    delta = np.inf  # initialize deviance to target correlation

    i = 1  # count tries

    while delta > tolerance:

        # Draw data from multivariate normal
        data = np.random.multivariate_normal(mean=mu, cov=sig, size=N)

        # Truncate data to limits
        if lim_a is not None:
            data[:, 0] = np.clip(data[:, 0], *lim_a)
        if lim_b is not None:
            data[:, 1] = np.clip(data[:, 1], *lim_b)

        # Compute generated correlation and deviance
        r_gen, p = pearsonr(data[:, 0], data[:, 1])
        delta = np.abs(r - r_gen)

        i += 1

    if verbose:
        print(f"Pearson r = {r_gen:.2f}, p = {p:.4f}")
        print(f"  (It took {i} tries to achieve the desired tolerance.)")

    return data
