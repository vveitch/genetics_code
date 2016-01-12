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

def beta_expectation(alpha_beta):
    """
    :param alpha_beta: shape (K,T,2) nparray where alpha_beta[k,t,0] = alpha_kt and alpha_beta[k,t,1]=beta_kt
    :return: a (K,T,2) nparray where r[k,t,0]=E(log theta_kt) and r[k,t,1]=E(log 1-theta_kt)
    """
    #note: seperated this out from diri_expect for ease of reading and 'cause I suspect it might be a source of bugs

    return psi(alpha_beta) - psi(np.sum(alpha_beta, 2))[:, :, np.newaxis]


class SVI_fixed_K_single_level:
    """
    Implements stochastic variational inference for genetics model
    with single level and fixed number of haplotypes
    """

    def __init__(self, K, T, N, alpha, beta, eta, tau0, kappa):
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
        self._T = T
        self._N = N

        #theta dist hyperparams
        self._alpha_beta = np.ones([self._K, self._T, 2])
        self._alpha_beta[:,:,0] = alpha*self._alpha_beta[:,:,0]
        self._alpha_beta[:,:,1] = beta*self._alpha_beta[:,:,1]
        #pi dist hyperparams
        self._eta = eta

        self._tau0 = tau0 + 1
        self._kappa = kappa

        # iteration counter, used for updating rho
        self._updatect = 0

        # Initialize the variational distribution q(pi|lambda)
        # todo: not totally sure this is a sensible initialization
        self._lambda = np.random.gamma(100., 1. / 100., self._K)
        self._E_log_pi = dirichlet_expectation(self._lambda)
        self._exp_E_log_pi = np.exp(self._E_log_pi)

        # initialize the variational distribution q(theta|gamma)
        # gamma_ktj = gamma[k,t,j] (up to indexing starting at 1 vs indexing starting at 0 anyways)

        # todo: adding a factor of 0.1 here means the haplotypes will be quite variable, which will cause the algorithm to start expressing a
        # strong preference in local and global variables right away... not sure if that's actually desirable
        self._gamma = np.random.gamma(100., 1. / 100, [self._K, self._T, 2])
        # the return of function has the structure E_logs_theta[k,t] = (E[log(theta_kt),E(log(1-theta_kt)])
        self._E_logs_theta = beta_expectation(self._gamma)

    def update_local(self, obs_snps):
        # return the sufficient stats needed to compute the global update

        batchN = len(obs_snps)

        # preallocate the variational distribution q(z1|phi1) and q(z2|phi2) for the mini-batch
        phi_1 = np.zeros((batchN, self._K))
        phi_2 = np.zeros((batchN, self._K))

        lambda_sstats = np.zeros(self._lambda.shape)
        gamma_sstats = np.zeros(self._gamma.shape)

        # shorthands
        lp = self._E_log_pi
        lts = self._E_logs_theta

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
            xi_n = np.random.beta(1, 1, sum(pa))  # set each relevant xi randomly
            # need to iterate to appx convergence between phase indicators c and probabilities phi

            for it in range(0, 100):
                #np.array makes these copies instead of just pointers
                last_phi1 = np.array(phi_1[n])
                last_phi2 = np.array(phi_2[n])

                # phase ambiguous terms
                phi_1[n] = np.exp(phi_n_base + np.sum(xi_n * lts[:, pa, 1], axis=1) \
                                  + np.sum((1 - xi_n) * lts[:, pa, 0], axis=1))
                phi_1[n] = phi_1[n] / sum(phi_1[n] + 1e-100)

                phi_2[n] = np.exp(phi_n_base + np.sum((1 - xi_n) * lts[:, pa, 1], axis=1) \
                                  + np.sum(xi_n * lts[:, pa, 0], axis=1))
                phi_2[n] = phi_2[n] / sum(phi_2[n] + 1e-100)

                xi_n = np.asarray(expit((phi_2[n] - phi_1[n]) * np.asmatrix((lts[:, pa, 0]) - (lts[:, pa, 1]))))
                # If phi hasn't changed much, we're done.
                meanchange = np.mean(abs(phi_1[n] - last_phi1)+abs(phi_2[n]-last_phi2))
                if (meanchange < 0.0001):
                    break

            lambda_sstats += (phi_1[n] + phi_2[n])

            # increment alpha parameter of beta dist when x_nj = 1
            # ([:,np.newaxis] for addition to each *column*)
            gamma_sstats[:, (xn == 0), 0] += (phi_1[n] + phi_2[n])[:, np.newaxis]
            gamma_sstats[:, (xn == 1), 0] += (np.outer(phi_2[n], xi_n) + np.outer(phi_1[n], (1 - xi_n)))

            # increment beta parameter of beta dist when x_nj = 0
            gamma_sstats[:, (xn == 2), 1] += (phi_1[n] + phi_2[n])[:, np.newaxis]
            gamma_sstats[:, (xn == 1), 1] += (np.outer(phi_1[n], xi_n) + np.outer(phi_2[n], (1 - xi_n)))

        return lambda_sstats, gamma_sstats

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

        # Do a local update to phi | lambda,gamma and xi | lambda, gamma
        # for this mini-batch. This also returns the information about phi and gamma that
        # we need to update lambda and gamma.
        (lambda_sstats, gamma_sstats) = self.update_local(observed_snps)

        # Update lambda
        self._lambda = self._lambda * (1 - rhos) + \
                       rhos * (self._eta + self._N / len(observed_snps) * lambda_sstats)
        self._E_log_pi = dirichlet_expectation(self._lambda)
        self._exp_E_log_pi = np.exp(self._E_log_pi)

        # Update gamma
        self._gamma = self._gamma * (1 - rhos) + \
                      rhos * (self._alpha_beta + self._N / len(observed_snps) * gamma_sstats)
        # the return of this function has the structure E_logs_theta[k,t] = (E[log(theta_kt),E(log(1-theta_kt)])
        self._E_logs_theta = beta_expectation(self._gamma)

        print np.max(self._E_logs_theta)
        print np.unravel_index(np.argmax(self._E_logs_theta),self._E_logs_theta.shape)

        slam = sorted(self._lambda,reverse=True) #for debugging

        # Update iteration counter
        self._updatect += 1


def main():
    np.random.seed(1)  # reproducibility

    K = 30
    T = 10
    N = 1000
    alpha = 0.1
    beta = 0.1
    eta = 0.1  # not sure about a good starter for this
    # copied from OnlineLDAVB example
    tau0 = 1
    kappa = 0.75

    data = np.load("simdata.npy")

    # #SVI
    # model = SVI_fixed_K_single_level(K,T,N,alpha,beta,eta,tau0,kappa)
    #
    # for i in range(0, N, 20):
    #     print i
    #     model.update_global(data[i:(i+19),:])
    #     print sorted(model._lambda,reverse=True)

    # batch VI by passing in same data set each time
    model = SVI_fixed_K_single_level(K=K, T=T, N=N, alpha=alpha, beta=beta, eta=eta, tau0=tau0, kappa=0)

    #batch
    for i in range(0, 500):
        print i
        if (i==7):
            print "for debugging"
        model.update_global(data)
 #       print sorted(model._lambda,reverse=True)


    np.save("SVI_output_batch",model._lambda)

if __name__ == '__main__':
    main()
