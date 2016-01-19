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
from scipy.special import expit as sp_expit  # inverse logit

#np.random.seed(100000001)
mean_change_thresh = 0.001

#changes default expit to prevent overflow errors
def expit(lp):
    tmp = np.array(lp) # just to make sure I'm not inadvertently messing up the lp value
    tmp[(lp>100)]=100
    tmp[(lp<-100)]=-100
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


def beta_expectation(alpha_beta):
    """
    :param alpha_beta: shape (K,T,2) nparray where alpha_beta[k,t,0] = alpha_kt and alpha_beta[k,t,1]=beta_kt
    :return: a (K,T,2) nparray where r[k,t,0]=E(log theta_kt) and r[k,t,1]=E(log 1-theta_kt)
    """
    # note: seperated this out from diri_expect for ease of reading and 'cause I suspect it might be a source of bugs

    return psi(alpha_beta) - psi(np.sum(alpha_beta, 2))[:, :, np.newaxis]


class SVI_fixed_K_single_level:
    """
    Implements stochastic variational inference for genetics model
    with single level and fixed number of haplotypes
    """

    def __init__(self, K, T, N, alpha, beta, eta, tau0, kappa, lambda_init=np.asarray([]), gamma_init=np.asarray([])):
        """
        Arguments:
        K: Number of haplotypes
        T: length of SNP sequence
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

        # theta dist hyperparams
        self._alpha_beta = np.ones([self._K, self._T, 2])
        self._alpha_beta[:, :, 0] = alpha * self._alpha_beta[:, :, 0]
        self._alpha_beta[:, :, 1] = beta * self._alpha_beta[:, :, 1]
        # pi dist hyperparams
        self._eta = eta

        self._tau0 = tau0 + 1
        self._kappa = kappa

        # iteration counter, used for updating rho
        self._updatect = 0

        # my best guess for a sensible intialization is a very large number of clusters
        # each with comparable probabilities and very heterogeneous haplotypes
        # OR: mostly homogeneous haplotypes coupled with pretty heterogeneous probabilities

        # Initialize the variational distribution q(pi|lambda)
        if (lambda_init.shape==(self._K,)):
            self._lambda = lambda_init
        else:
            # todo: not totally sure this is a sensible initialization
            # (2000,1./100) gives all probabilities within 1 order of magnitude of each other
            self._lambda = np.random.gamma(100, 1. / 100, self._K)
        self._E_log_pi = dirichlet_expectation(self._lambda)
        self._exp_E_log_pi = np.exp(self._E_log_pi)

        # initialize the variational distribution q(theta|gamma)
        # gamma_ktj = gamma[k,t,j] (up to indexing starting at 1 vs indexing starting at 0 anyways)
        if (gamma_init.shape==(self._K, self._T, 2)):
            self._gamma = gamma_init
        else:
            # TODO: figure out good starting value for this (and why 1,1 causes immediate crash)
            self._gamma = np.random.gamma(10, 1. / 100, [self._K, self._T, 2])
            #self._gamma = np.random.gamma(10,1./10,[self._K, self._T, 2])
        # the return of function has the structure E_logs_theta[k,t] = (E[log(theta_kt),E(log(1-theta_kt)])
        self._E_logs_theta = beta_expectation(self._gamma)

    def update_local_no_phase_ambig(self, obs_snps):
        # return the sufficient stats needed to compute the global update, as computed by
        # ignoring any contributions from sites with ambiguous phase (minor allele count =1)
        # with enough data this works pretty well, and is very fast
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

            # contribution for phase unambiguous terms is same for both hap indicators
            phi_n_base = lp + np.sum(lts[:, (xn == 0), 1], axis=1) + np.sum(lts[:, (xn == 2), 0], axis=1)

            phi_1[n] = np.exp(phi_n_base) + 1e-100
            phi_1[n] = phi_1[n] / sum(phi_1[n])

            phi_2[n] = np.exp(phi_n_base) + 1e-100
            phi_2[n] = phi_2[n] / sum(phi_2[n])

            lambda_sstats += (phi_1[n] + phi_2[n])

            # increment alpha parameter of beta dist when x_nj = 1
            # ([:,np.newaxis] for addition to each *column*)
            gamma_sstats[:, (xn == 0), 0] += (phi_1[n] + phi_2[n])[:, np.newaxis]

            # increment beta parameter of beta dist when x_nj = 0
            gamma_sstats[:, (xn == 2), 1] += (phi_1[n] + phi_2[n])[:, np.newaxis]

        return lambda_sstats, gamma_sstats

    def prob_c(self, c, phi_n_base, pa):
        """
        helper function to compute probabilities of phase indicators
        :argument:
        c: the bit string
        phi_n_base: the phi component that doesn't depend on phase ambiguous terms
        :return:
        the probability of the bit string c
        """
        lts = self._E_logs_theta
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

            ####PHASE AMBIGUOUS DATA INCLUDED####
            # this approach takes an exact sample of the phase indicators, if ever this code gets used for really large T
            # then maybe this should be switched to a gibbs sampler or something

            # todo: the problem is that the expected value of c is approx 0.5 pretty often, which leads to dilution of
            # the haplotypes, triggering a feedback that assigns everybody to a single overwhelmingly popular haplotype with
            # neutral probabilities

            pa = (xn == 1)  # shorthand for index of sites with ambiguous phases
            c = np.repeat(0.5,sum(pa)) #doesn't really work

            # all of this stuff was nonsense, but there's definitely the nexus of a good idea here
            # if (sum(pa)==0):
            #     c=np.array([])
            # else:
            #     # collection of all possible phase indicators
            #     cs = np.zeros([sum(pa), 2])
            #     cs[:, 1] += 1
            #     cs = cartesian(cs)
            #
            #     # sample the value of c
            #     # todo: I could also compute an exact average using this...
            #     phase_probs = map(partial(self.prob_c, phi_n_base=phi_n_base, pa=pa), cs)
            #     # todo: vv: this shouldn't be necessary... either I made a mistake or it's because of expit overflow errors
            #     phase_probs = phase_probs / sum(phase_probs)
            #     #c = cs[np.random.multinomial(1, phase_probs).nonzero()[0][0]]
            #     c = np.asarray(phase_probs*np.asmatrix(cs)) #exact expectation

            # update the values of phi
            phi_1_n = np.exp(phi_n_base + np.sum(c * lts[:, pa, 1], axis=1) \
                             + np.sum((1 - c) * lts[:, pa, 0], axis=1)) + 1e-100
            phi_1_n = phi_1_n / sum(phi_1_n)

            phi_2_n = np.exp(phi_n_base + np.sum((1 - c) * lts[:, pa, 1], axis=1) \
                             + np.sum(c * lts[:, pa, 0], axis=1)) + 1e-100
            phi_2_n = phi_2_n / sum(phi_2_n)

            lambda_sstats += (phi_1_n + phi_2_n)

            # increment alpha parameter of beta dist when x_nj = 1
            # ([:,np.newaxis] for addition to each *column*)
            gamma_sstats[:, (xn == 0), 0] += (phi_1_n + phi_2_n)[:, np.newaxis]
            gamma_sstats[:, (xn == 1), 0] += (np.outer(phi_2_n, c) + np.outer(phi_1_n, (1 - c)))

            # increment beta parameter of beta dist when x_nj = 0
            gamma_sstats[:, (xn == 2), 1] += (phi_1_n + phi_2_n)[:, np.newaxis]
            gamma_sstats[:, (xn == 1), 1] += (np.outer(phi_1_n, c) + np.outer(phi_2_n, (1 - c)))

            # #faster VI approach. Unfortunately seems too sensitive to initial setting of xi_n
            # xi_n = np.random.beta(1, 1, sum(pa))  # set each relevant xi randomly
            # # need to iterate to appx convergence between phase indicators c and probabilities phi
            #
            # for it in range(0, 100):
            #     # np.array makes these copies instead of just pointers
            #     last_phi1 = np.array(phi_1[n])
            #     last_phi2 = np.array(phi_2[n])
            #
            #     # phase ambiguous terms
            #     phi_1[n] = np.exp(phi_n_base + np.sum(xi_n * lts[:, pa, 1], axis=1) \
            #                       + np.sum((1 - xi_n) * lts[:, pa, 0], axis=1)) + 1e-100
            #     phi_1[n] = phi_1[n] / sum(phi_1[n])
            #
            #     phi_2[n] = np.exp(phi_n_base + np.sum((1 - xi_n) * lts[:, pa, 1], axis=1) \
            #                       + np.sum(xi_n * lts[:, pa, 0], axis=1)) + 1e-100
            #     phi_2[n] = phi_2[n] / sum(phi_2[n])
            #
            #     xi_n = (phi_2[n] - phi_1[n]) * np.asmatrix((lts[:, pa, 0]) - (lts[:, pa, 1]))
            #     xi_n = np.asarray(expit(xi_n))
            #
            #     # If phi hasn't changed much, we're done.
            #     meanchange = np.mean(abs(phi_1[n] - last_phi1)+abs(phi_2[n]-last_phi2))
            #     if (meanchange < 0.000001):
            #         break

            # lambda_sstats += (phi_1[n] + phi_2[n])
            #
            # # increment alpha parameter of beta dist when x_nj = 1
            # # ([:,np.newaxis] for addition to each *column*)
            # gamma_sstats[:, (xn == 0), 0] += (phi_1[n] + phi_2[n])[:, np.newaxis]
            # gamma_sstats[:, (xn == 1), 0] += (np.outer(phi_2[n], xi_n) + np.outer(phi_1[n], (1 - xi_n)))
            #
            # # increment beta parameter of beta dist when x_nj = 0
            # gamma_sstats[:, (xn == 2), 1] += (phi_1[n] + phi_2[n])[:, np.newaxis]
            # gamma_sstats[:, (xn == 1), 1] += (np.outer(phi_1[n], xi_n) + np.outer(phi_2[n], (1 - xi_n)))

        return lambda_sstats, gamma_sstats

    def update_global(self, observed_snps,include_phased=True):
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
        # for this mini-batch. This also returns the information about phi and gamma that
        # we need to update lambda and gamma.

        # (lambda_sstats, gamma_sstats) = self.update_local(observed_snps)


        #idea: run an initial fit using phase data, but then rejigger the starting haplotypes to
        # reflect options for all the phase ambiguous sites (which might increase the candidate pool by a factor of
        # 32ish).

        # to avoid phase ambig terms swamping out probability updates, do the first n iterations
        # using no phase ambig data - this should give a reasonable jumping off point for theta values
        # if (self._updatect <= 50):
        #     (lambda_sstats, gamma_sstats) = self.update_local_no_phase_ambig(observed_snps)
        # else:
        #     (lambda_sstats, gamma_sstats) = self.update_local(observed_snps)

        if (include_phased):
            (lambda_sstats, gamma_sstats) = self.update_local(observed_snps)
        else:
            #update ignoring phased data
            (lambda_sstats, gamma_sstats) = self.update_local_no_phase_ambig(observed_snps)


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

        # debugging
        slam1=np.asarray(sorted(lambda_sstats,reverse=True))
        sgam1=np.asarray([x for (y,x) in sorted(zip(lambda_sstats,gamma_sstats), reverse=True, key=lambda pair: pair[0])])

        # print np.max(self._E_logs_theta)
        # print np.unravel_index(np.argmax(self._E_logs_theta),self._E_logs_theta.shape)


        # Update iteration counter
        self._updatect += 1


def main():
#    np.random.seed(1)  # reproducibility

    K = 100
    T = 10
    N = 1000
    alpha = 0.1
    beta = 0.1
    eta = 1  # not sure about a good starter for this
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

    ###batch VI
    # set kappa=0 (so rhos=1 always) and pass in same data set each time
    model = SVI_fixed_K_single_level(K=K, T=T, N=N, alpha=alpha, beta=beta, eta=eta, tau0=tau0, kappa=0)

    # get a reasonable starting position in the landscape by first estimating proportions and haplotypes ignoring the
    # data with ambiguous phases... this converges very quickly
    for i in range(0, 100):
        if (np.mod(i, 10) == 0):
            print i
            print sorted(model._lambda, reverse=True)
        model.update_global(data,include_phased=False)

    #this model now has a pretty good idea about sites that can be estimated from phase unambig data
    #(sgam[k,t,a],sgam[k,t,b]) are the (alpha,beta) parameters of the posterior beta distribution of theta_kt
    # if alpha/beta is far from 1 then we have strong evidence for either 1 or a 0 at this site
    # if alpha/beta is approximately 1 then there are likely multiple haplotypes that differ at this site but are
    # otherwise consistent.
    # idea: make a starting position using this information by preserving strong ratios and "forking" weak ones into new
    # candidate haplotypes.

    #order both gamma values and lambda values by the lambda weights
    slam=np.asarray(sorted(model._lambda,reverse=True))
    sgam=np.asarray([x for (y,x) in sorted(zip(model._lambda,model._gamma), reverse=True, key=lambda pair: pair[0])])

    #only the haplotypes with non-trivial probability should be kept
    slam=slam[(slam>1.5*eta)]
    sgam=sgam[0:len(slam)]

    lambda_start=np.empty([0])
    gamma_start=np.empty([0,T,2])

    #iterate through gammas with non-trivial probability
    for i in range(0,len(sgam)):
        #index of terms where alpha/beta is "close" to 1
        ambig_index=(abs(np.log(sgam[i,:,0] / sgam[i,:,1])/np.log(10))<1)
        #emphasize discovered ratio
        new_gam = 10*pow(np.transpose(np.transpose(sgam[i,:,:]) / np.sum(sgam[i,:,:],axis=1)),2)

        if (sum(ambig_index)==0):
            gamma_start = np.append(gamma_start,new_gam[np.newaxis,:,:],0)
            lambda_start = np.append(lambda_start,slam[i])
        else:
            #iterate through ambiguous indices
            for j in range(0,pow(2,sum(ambig_index))):
                #j in binary
                b_string=np.unravel_index(j,2*np.ones(sum(ambig_index),dtype=int))
                #different ratios of alpha/beta
                new_vals=np.asarray(map(lambda b: [0.1,9.9] if b == 0 else [9.9,0.1], b_string))
                new_gam[ambig_index]=new_vals
                gamma_start = np.append(gamma_start,new_gam[np.newaxis,:,:],0)
                lambda_start = np.append(lambda_start,slam[i]) #same relative probability for all forks from same base gamma

    #append some "flat" haplotypes with low probability to catch misidentified cases
    #idea is llhd from terms with good fit should swamp out these
    #but risk is that these stay flat and end up eating all the probability
    gamma_start = np.append(gamma_start,np.random.gamma(50, 1. / 100, [10, T, 2]),0)
    lambda_start = np.append(lambda_start,np.repeat(slam[i],10))

    lambda_start = np.repeat(1,len(lambda_start)) #testing
#    lambda_start = len(lambda_start)/sum(lambda_start) * lambda_start #avg value = 1
    gamma_start = gamma_start #multiplying this weights the initial biases higher


    np.save("starter_lambda",lambda_start)
    np.save("starter_gamma",gamma_start)

    lambda_st = np.load("starter_lambda.npy")
    gamma_st = np.load("starter_gamma.npy")

    K=len(lambda_st)

    model = SVI_fixed_K_single_level(K,T,N,alpha,beta,eta,tau0,kappa,lambda_init=lambda_st,gamma_init=gamma_st)

    #large K makes this slow... but just a few iterations will throw away almost all of the clusters
    for i in range(0,10):
        if (np.mod(i, 0) == 0):
            print i
            print sorted(model._lambda, reverse=True)
        model.update_global(data)

    #order both gamma values and lambda values by the lambda weights
    slam=np.asarray(sorted(model._lambda,reverse=True))
    sgam=np.asarray([x for (y,x) in sorted(zip(model._lambda,model._gamma), reverse=True, key=lambda pair: pair[0])])

    #only the haplotypes with non-trivial probability should be kept
    slam=slam[(slam>1.1*eta)]
    sgam=sgam[0:len(slam)]

    reduced_lambda = slam
    reduced_gamma = sgam

    np.save("reduced_lambda",reduced_lambda)
    np.save("reduced_gamma",reduced_gamma)

    K=len(reduced_lambda)
    model = SVI_fixed_K_single_level(K,T,N,alpha,beta,eta,tau0,kappa,lambda_init=reduced_lambda,gamma_init=reduced_gamma)

    # batch
    for i in range(0, 1001):
        if (np.mod(i, 5) == 0):
            print i
            print sorted(model._lambda, reverse=True)
        if (np.mod(i,100)==0):
            np.save("lambda"+str(i), model._lambda)
            np.save("gamma"+str(i), model._gamma)
        model.update_global(data)

if __name__ == '__main__':
    main()
