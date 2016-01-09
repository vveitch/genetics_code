# Victor Veitch
# 01-06-2016
# This fits the genome model assuming a fixed number of haplotypes
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

import sys
import numpy as np
from scipy.special import psi  # digamma
from scipy.special import expit  # inverse logit

np.random.seed(100000001)
mean_change_thresh = 0.001


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
    return psi(alpha) - psi(np.sum(alpha, 2))[:, :, np.newaxis]


class SVI_fixed_K_single_level:
    """
    Implements stochastic variational inference for genetics model
    with single level and fixed number of haplotypes
    """

    def __init__(self, K, N, alpha, beta, eta, tau0, kappa):
        """
        Arguments:
        K: Number of haplotypes
        N: Total number of people in the population.
        alpha: Hyperparameter for prior on haplotypes theta
        beta: Hyperparameter for prior on haplotypes theta
        eta: Hyperparameter for prior on haplotype weights pi
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.

        todo: write equiv statement for genetics model
        Note that if you pass the same set of D documents in every time and
        set kappa=0 this class can also be used to do batch VB.
        """

        self._K = K
        self._N = N

        self._alpha = alpha
        self._beta = beta
        self._eta = eta

        self._tau0 = tau0 + 1
        self._kappa = kappa

        # iteration counter, used for updating rho
        self._updatect = 0

        # Initialize the variational distribution q(pi|lambda)
        # todo: not totally sure this is a sensible initialization
        # roughly this is all clusters have same expected weight, but w very high variance
        # max(dirichlet(lambda)) tends to be 15-30%
        # this does reflect what I think reality is,
        # but maybe it starts the search at a nasty point in the parameter landscape
        self._lambda = 0.1 * np.random.gamma(100., 1. / 100., self._K)
        self._E_log_pi = dirichlet_expectation(self._lambda)
        self._exp_E_log_pi = np.exp(self._Elogbeta)

        # initialize the variational distribution q(theta|gamma)
        # gamma_ktj = gamma[k,t,j] (up to indexing starting at 1 vs indexing starting at 0 anyways)
        self._gamma = 0.1 * np.random.gamma(100., 1. / 100, [self._K, self._T, 2])
        # the return of function has the structure E_logs_theta[k,t] = (E[log(theta_kt),E(log(1-theta_kt)])
        self._E_logs_theta = dirichlet_expectation(self._gamma)

    def update_local(self, obs_snps):
        # return the sufficient stats needed to compute the global update

        batchN = len(obs_snps)

        # Initialize the variational distribution q(z|phi) for the mini-batch
        phi = 1 * np.random.dirichlet(np.ones(self._K), (batchN, self._K))

        # Initialize the variational distribution q(c|xi) for the mini-batch
        # note that these parameters are only relevant when obs_snps[n,t]=1
        # so much of this will be irrelevant
        xi = 1 * np.random.beta(1, 1, (batchN, self._T))

        lambda_sstats = np.zeros(self._lambda.shape)
        gamma_sstats = np.zeros(self._gamma.shape)

        # shorthands
        lp = self._E_log_pi
        lts = self._E_logs_theta

        # Now, for each person n update that person's phi and xi
        it = 0
        meanchange = 0
        for n in range(0, batchN):

            xn = obs_snps[n]
            # (xn==j) gives sites where minor allele count is j
            # todo: rewrite tex equation to look more obviously like how I coded this
            # todo: did I really do this right? the idea is to compute the appropriate equation for all k at once in a vectorized style
            # contribution for phase unambiguous terms is same for both hap indicators
            phi_n_base = lp + np.sum(lts[:, (xn == 0), 1], axis=1) + np.sum(lts[:, (xn == 2), 0], axis=1)

            pa = (xn == 1) # shorthand for index of sites with ambiguous phases
            #need to iterate to appx convergence between phase indicators c and probabilities phi

            for it in range(0,100):
                # phase ambiguous terms
                phi_n1 = np.exp(phi_n_base + np.sum(xi_n[:, pa] * lts[:, pa, 1], axis=1) \
                            + np.sum((1 - xi_n[:, pa]) * lts[:, pa, 0], axis=1))

                phi_n2 = np.exp(phi_n_base + np.sum((1 - xi_n[:, pa]) * lts[:, pa, 1], axis=1) \
                            + np.sum(xi_n[:, pa] * lts[:, pa, 0], axis=1))
                xi_n = expit()#todo

            print sum(wordcts[d])
            # These are mostly just shorthand (but might help cache locality)
            ids = wordids[d]
            cts = wordcts[d]
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self._expElogbeta[:, ids]
            # The optimal phi_{dwk} is proportional to
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
            # Iterate between gamma and phi until convergence
            for it in range(0, 100):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self._alpha + expElogthetad * \
                                       np.dot(cts / phinorm, expElogbetad.T)
                print gammad[:, np.newaxis]
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = np.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):
                    break
            gamma[d, :] = gammad
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats[:, ids] += np.outer(expElogthetad.T, cts / phinorm)

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk}
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self._expElogbeta

        return ((gamma, sstats))

    def update_global(self, observed_snps):
        """
        Takes in a minibatch of data, does a local update and then
        uses the result of that update to update the variational
        parameters for the global variables (q(pi|lambda) & q(theta|gamma))

       observed_snps:  numpy array of n people at t contiguous sites.
        Each observation is represented as a list of minor allele counts

        """

        # rhos will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhos = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhos = rhos

        # Do a local update to phi | lambda,gamma and xi | lambda, gamma
        # for this mini-batch. This also returns the information about phi and gamma that
        # we need to update lambda and gamma.
        (lambda_sstats, gamma_sstats) = self.update_local(observed_snps)

        # Update lambda
        self._lambda = self._lambda * (1 - rhos) + \
                       rhos * (self._eta + self._N / len(observed_snps) * lambda_sstats)
        self._E_log_pi = dirichlet_expectation(self._lambda)
        self._exp_E_log_pi = np.exp(self._Elogbeta)

        # Update gamma
        self._gamma = self._gamma * (1 - rhos) + rhos * (gamma_sstats)
        # the return of this function has the structure E_logs_theta[k,t] = (E[log(theta_kt),E(log(1-theta_kt)])
        self._E_logs_theta = dirichlet_expectation(self._gamma)

        # Update iteration counter
        self._updatect += 1


def main():
    infile = sys.argv[1]
    K = int(sys.argv[2])
    alpha = float(sys.argv[3])
    eta = float(sys.argv[4])
    kappa = float(sys.argv[5])
    S = int(sys.argv[6])

    docs = corpus.corpus()
    docs.read_data(infile)

    vocab = open(sys.argv[7]).readlines()
    model = OnlineLDA(vocab, K, 100000,
                      0.1, 0.01, 1, 0.75)
    for i in range(1000):
        print i
        wordids = [d.words for d in docs.docs[(i * S):((i + 1) * S)]]
        wordcts = [d.counts for d in docs.docs[(i * S):((i + 1) * S)]]
        model.update_lambda(wordids, wordcts)
        np.savetxt('/tmp/lambda%d' % i, model._lambda.T)


# infile = open(infile)
#     corpus.read_stream_data(infile, 100000)

if __name__ == '__main__':
    main()
