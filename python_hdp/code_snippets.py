__author__ = 'victor'

import numpy as np
from scipy.special import psi  # digamma
from scipy.special import expit as sp_expit  # inverse logit
from functools import partial
from sklearn.utils.extmath import cartesian

#sort gamma and weights the same
lam1=np.load("SVI_output_batch_nophase_lambda_1.npy")
gam1=np.load("SVI_output_batch_nophase_gamma_1.npy")
slam1=np.asarray(sorted(lam1,reverse=True))
sgam1=np.asarray([x for (y,x) in sorted(zip(lam1,gam1), reverse=True, key=lambda pair: pair[0])])

#for the version with every haplotype
T=10
theta = cartesian(np.repeat(np.array([[0.001,0.999]]),T,0))
l50 = np.load("lambda_all_haps5.npy")
sl50 = np.asarray(sorted(l50,reverse=True))
st=np.asarray([x for (y,x) in sorted(zip(l50,theta), reverse=True, key=lambda pair: pair[0])])

#for the simulated data
w = np.load("simweights.npy")
haps = np.load("simhaps.npy")
sw = np.asarray(sorted(w,reverse=True))
shaps=np.asarray([x for (y,x) in sorted(zip(w,haps), reverse=True, key=lambda pair: pair[0])])

#check how many times some haplotype in the fit appears in top 20 of actual haplotypes
sum([np.array_equal(a,np.round(st[5])) for a in np.round(shaps)[0:20]])

#llhd of data with parameter pi