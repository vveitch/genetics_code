# Victor Veitch
# 01-06-2016
# This fits the genome model using all possible haplotypes
# some of the code I'm using here is adapted from Matt Hoffman's
# onlineldavb.py package for fitting LDA.
#
#
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from scipy.special import psi  # digamma
from scipy.special import expit as sp_expit  # inverse logit
from functools import partial
from sklearn.utils.extmath import cartesian

#np.random.seed(100000001)
mean_change_thresh = 0.001

#changes default expit to prevent overflow errors
def expit(lp):
    tmp = lp
    tmp[(lp>100)]=100
    tmp[(lp<int(-100))]=int(-100)
    return sp_expit(tmp)


def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return (psi(alpha) - psi(np.sum(alpha)))
    # in this case each alpha[k] is a diri parameter and return is appropriate matrix
    elif (len(alpha.shape) == 2):
        return psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]
        # case: each alpha[k,j] is a diri parameter and return is appropriate tensor


class SVI_fixed_K_single_level:
    """
    Implements stochastic variational inference for genetics model
    with single level and fixed number of haplotypes
    """

    def __init__(self, T, N, eta, tau0, kappa, lambda_init=np.asarray([])):
        """
        Arguments:
        K: Number of haplotypes
        T: Length of SNP sequence
        N: Total number of people in the population.
        eta: Hyperparameter for prior on haplotype weights pi
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.

        Note that if you pass the same data in every time and
        set kappa=0 this class can also be used to do batch VB.
        """

        self._K = pow(2,T)
        self._T = T
        self._N = N

        # pi dist hyperparams
        self._eta = eta

        self._tau0 = tau0 + 1
        self._kappa = kappa

        # iteration counter, used for updating rho
        self._updatect = 0

        # Initialize the variational distribution q(pi|lambda)
        if (lambda_init.shape==(self._K,)):
            self._lambda = lambda_init
        else:
            # todo: not totally sure this is a sensible initialization
            # (2000,1./100) gives all probabilities within 1 order of magnitude of each other
            self._lambda = np.random.gamma(100, 1. / 100, self._K)
        self._E_log_pi = dirichlet_expectation(self._lambda)
        self._exp_E_log_pi = np.exp(self._E_log_pi)

        #all theta values
        theta = cartesian(np.repeat(np.array([[0.0001,0.9999]]),T,0))
        self.logs_theta = np.zeros([self._K, self._T, 2])
        self.logs_theta[:,:,0] = np.log(theta)
        self.logs_theta[:,:,1] = np.log(1-theta)

    def prob_c(self, c, phi_n_base, pa):
        """
        helper function to compute probabilities of phase indicators
        :argument:
        c: the bit string
        phi_n_base: the phi component that doesn't depend on phase ambiguous terms
        pa: bool array indicating the phase ambiguous terms
        :return:
        the probability of the bit string c
        """
        lts = self.logs_theta
        phi_1 = np.exp(phi_n_base + np.sum(c * lts[:, pa, 1], axis=1) \
                       + np.sum((1 - c) * lts[:, pa, 0], axis=1)) + 1e-100
        phi_1 = phi_1 / sum(phi_1)

        phi_2 = np.exp(phi_n_base + np.sum((1 - c) * lts[:, pa, 1], axis=1) \
                       + np.sum(c * lts[:, pa, 0], axis=1)) + 1e-100
        phi_2 = phi_2 / sum(phi_2)

        # probability of the string is product of the probabilities at each site
        return np.prod(expit((phi_2 - phi_1) * np.asmatrix((lts[:, pa, 0]) - (lts[:, pa, 1]))))

    def update_local(self, obs_snps):
        # return the sufficient stats needed to compute the global update

        batchN = len(obs_snps)

        lambda_sstats = np.zeros(self._lambda.shape)

        # shorthands
        lp = self._E_log_pi
        lts = self.logs_theta

        # Now, for each person n update that person's phi and xi
        # meanchange = 0
        for n in range(0, batchN):
            # data for nth person
            # vv: the asarray here should be redundant, but pycharm was giving me grief for omitting it
            xn = np.asarray(obs_snps[n])

            # (xn==j) gives sites where minor allele count is j
            # todo: rewrite tex equation to look more obviously like how I coded this
            # todo: did I really do this right? the idea is to compute the appropriate equation for all k at once in a vectorized style

            # contribution for phase unambiguous terms is same for both hap indicators
            phi_n_base = lp + np.sum(lts[:, (xn == 0), 1], axis=1) + np.sum(lts[:, (xn == 2), 0], axis=1)

            pa = (xn == 1)  # shorthand for index of sites with ambiguous phases
            c = np.repeat(0.5,sum(pa))
            # if (sum(pa)==0):
            #     c=np.array([])
            # else:
            #     # collection of all possible phase indicators
            #     cs = np.zeros([sum(pa), 2])
            #     cs[:, 1] = 1
            #     cs = cartesian(cs)
            #
            #     #exact expectation of c
            #     phase_probs = map(partial(self.prob_c, phi_n_base=phi_n_base, pa=pa), cs)
            #     # todo: vv: this shouldn't be necessary... either I made a mistake or it's because of expit overflow errors
            #     phase_probs = phase_probs / sum(phase_probs)
            #     #c = cs[np.random.multinomial(1, phase_probs).nonzero()[0][0]] #random sample
            #     c = np.asarray(phase_probs*np.asmatrix(cs)) #exact expectation

            # update the values of phi
            phi_1_n = np.exp(phi_n_base + np.sum(c * lts[:, pa, 1], axis=1) \
                             + np.sum((1 - c) * lts[:, pa, 0], axis=1)) + 1e-100
            phi_1_n = phi_1_n / sum(phi_1_n)

            phi_2_n = np.exp(phi_n_base + np.sum((1 - c) * lts[:, pa, 1], axis=1) \
                             + np.sum(c * lts[:, pa, 0], axis=1)) + 1e-100
            phi_2_n = phi_2_n / sum(phi_2_n)

            lambda_sstats += (phi_1_n + phi_2_n)

        return lambda_sstats

    def update_global(self, observed_snps):
        """
        Takes in a minibatch of data, does a local update and then
        uses the result of that update to update the variational
        parameters for the global variables (q(pi|lambda) & q(theta|gamma))

        observed_snps:  numpy array of n people at t contiguous sites.
        Each observation is represented as a list of minor allele counts
        include_phased: bool indicating whether likelihood contribution from sites with ambiguous phase should be included
        """

        # rhos will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhos = pow(self._tau0 + self._updatect, -self._kappa)

        # Do a local update to phi | lambda,gamma and xi | lambda, gamma
        # for this mini-batch. This also returns the information about phi
        # we need to update lambda

        lambda_sstats = self.update_local(observed_snps)

        # Update lambda
        self._lambda = self._lambda * (1 - rhos) + \
                       rhos * (self._eta + self._N / len(observed_snps) * lambda_sstats)
        self._E_log_pi = dirichlet_expectation(self._lambda)
        self._exp_E_log_pi = np.exp(self._E_log_pi)

        # debugging
        # slam1=np.asarray(sorted(lambda_sstats,reverse=True))
        # sgam1=np.asarray([x for (y,x) in sorted(zip(lambda_sstats,gamma_sstats), reverse=True, key=lambda pair: pair[0])])

        # print np.max(self._E_logs_theta)
        # print np.unravel_index(np.argmax(self._E_logs_theta),self._E_logs_theta.shape)


        # Update iteration counter
        self._updatect += 1


def main():
#    np.random.seed(1)  # reproducibility

    T = 10
    N = 1000
    eta = 1./pow(2,T)  # not sure about a good starter for this
    tau0 = 1
    kappa = 0.75

    data = np.load("simdata.npy")

    ###batch VI
    # set kappa=0 (so rhos=1 always) and pass in same data set each time
    model = SVI_fixed_K_single_level(T=T, N=N, eta=eta, tau0=tau0, kappa=0)

    # batch
    for i in range(0, 1001):
        if (np.mod(i, 1) == 0):
            print i
            print np.sort(model._lambda)[::-1]
        if (np.mod(i,5)==0):
            np.save("lambda_all_haps"+str(i), model._lambda)
        model.update_global(data)

if __name__ == '__main__':
    main()
