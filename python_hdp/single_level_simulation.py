# Victor Veitch
# 01-01-2016
# draws a realization of a chunk of the genome of a population according to
# my modification of Xing et al. DP model

# todo: maybe make a person class that tracks the allele ids as well for convenience of testing
# although I really only care about how accurately I can estimate the weights anyways

import numpy as np
import scipy.stats as sp


def draw_stick_breaking(alpha=1, max_atoms=1000, cutoff=1e-10):
    # returns (truncated) list of DP atom probabilities
    # arguments:
    # alpha: concentration parameter of DP
    # max_atoms: maximum possible number of atoms
    # cutoff: smallest possible atom probability

    list_dim = 0
    rest = 1
    beta_list = sp.beta(1, alpha).rvs(size=max_atoms)
    sigma_list = np.zeros(max_atoms)

    for k in range(max_atoms):
        sigma_list[k] = beta_list[k] * rest
        rest *= (1 - beta_list[k])
        if rest < cutoff:
            list_dim = k
            break

    return sigma_list[0:list_dim]


def draw_haplotype(alpha, beta, hap_length):
    # returns a vector of probabilities, each of which is probability of allele=1 at that site
    # warning: standard is to code minor allele as 1, but I'm not imposing that the "1" allele should
    # be the less common one at the site
    # arguments:
    # alpha: shape parameter for each site
    # beta: shape parameter for each site
    # hap_length: maximum possible number of atoms
    #
    # todo: allow for distinct shape parameters at each site (to reflect empirically observed freqs)
    # todo: maybe allow some prior at each site

    return sp.beta(alpha, beta).rvs(size=hap_length)


def draw_haplotypes(num_draws, alpha, beta, hap_length):
    # returns a matrix, each row of which is a (realization of) a haplotype.
    # arguments:
    # num_draws: number of haplotypes (atom locations) to draw
    # alpha: shape parameter for each site
    # beta: shape parameter for each site
    # hap_length: maximum possible number of atoms
    #
    # todo: allow for distinct shape parameters at each site (to reflect empirically observed freqs)
    # todo: maybe allow some prior at each site

    haps = np.zeros(shape=[num_draws, hap_length])
    for k in range(num_draws):
        haps[k,] = draw_haplotype(alpha, beta, hap_length)

    return haps


def draw_person(weights, haplotypes):
    # returns a list where each entry is number of minor alleles at that site (0,1 or 2)
    # arguments:
    # weights: truncated list of DP weights
    # haplotypes: list of haplotypes (same length as list of weights)

    hap_length = haplotypes[0,].size

    # iterate over sites
    for k in range(hap_length):
        # haplotype indices
        z1 = np.random.multinomial(1, weights).nonzero()[0][0]
        z2 = np.random.multinomial(1, weights).nonzero()[0][0]

        # alleles
        a1 = [np.random.binomial(1, p) for p in haplotypes[z1,]]
        a2 = [np.random.binomial(1, p) for p in haplotypes[z2,]]

        # forget the phase by taking cannonical order as (0,1)
        # return map(lambda pair: (pair[1], pair[0]) if pair[0] == 1 else pair, zip(a1, a2))

        # forget the phase by just returning the sum of the minor alleles
        return map(sum, zip(a1, a2))


def draw_people(num_people, weights, haplotypes):
    # returns list of people
    # arguments:
    # weights: truncated list of DP weights
    # haplotypes: list of haplotypes (same length as list of weights)
    return [draw_person(weights, haplotypes) for _ in range(num_people)]


def main():
    np.random.seed(1)  # reproducibility
    # todo: ask wei what she thinks about realistic parameters for DP and haplotypes

    weights = draw_stick_breaking(5, 1000, 1e-10)
    # setting alpha=beta=small gives values close to either 0 or 1 whp,
    # which matches my intuition for how these oughta look
    haplotypes = draw_haplotypes(weights.size, 0.1, 0.1, 10)
    people = draw_people(1000, weights, haplotypes)
    people = np.asarray(people)

    print(people[0:5]).shape[0]

    # ballparking frequencies
    print sum((people==0))
    print sum((people==1))
    print sum((people==2))

    # print haplotypes
    # print weights
    # with concentration parameter = 5 for the DP looks like top 20 entries consistently have combined prob >95%
    print sum(sorted(weights,reverse=True)[0:20])

if __name__ == '__main__':
    main()
