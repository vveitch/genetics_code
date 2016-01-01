import numpy as np
import scipy.stats as sp


def draw_stick_breaking(alpha=1, T=1000, cutoff=1e-10):
    # returns (truncated) list of DP atom probabilities
    # arguments:
    # alpha: concentration parameter of DP
    # T: maximum possible number of atoms
    # cutoff: smallest possible atom probability
    beta_prime_ks = []

    list_dim = 0
    rest = 1
    beta_list = sp.beta(1, alpha).rvs(size=T)
    sigma_list = np.zeros(T)

    for k in range(T):
        sigma_list[k] = beta_list[k] * rest
        rest *= (1 - beta_list[k])
        if rest < cutoff:
            list_dim = k
            break

    return sigma_list[0:list_dim]


def main():
 #   np.random.seed(1)  # reproducibility
    print (draw_stick_breaking()).size

if __name__ == '__main__':
    main()
