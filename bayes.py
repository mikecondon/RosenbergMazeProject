import numpy as np
from matplotlib import pyplot as plt
from scipy.special import factorial, loggamma
from scipy.stats import poisson
from tqdm import tqdm


def P(t, s, arr, log=False, normalise=False, alpha=0.5, beta=0.5):
	y = arr[t:s+1]
	y_sum = np.sum(y)
	numer = loggamma(alpha+y_sum) + alpha*np.log(beta) 
	denom = loggamma(alpha) + (alpha+y_sum-1)*np.log(beta+s-t+1)
	log_P = numer - denom
	if normalise:
		norm =  np.sum(np.log(factorial(y)))
		log_P -= norm
	if log:
		return log_P
	return np.exp(log_P)

def pi_tau_m(m, n, tau, log=False):
	if log:
		return np.log(2*m) + np.log(2*m+1) - (2*m+1)*np.log(n) + (2*m-1)*np.log(x)+np.log(n-x)
	return (2*m+1)*2*m*(n-tau)*tau**(2*m-1) / (n**(2*m+1))

def pi_tau(t, s):
	return s - t - 1

def Q(m, j, t, arr, log=False):
	n = len(arr)
	if j==m:
		if log:
			return P(t, n, arr, True)+np.log(pi_tau_m(m, n, t-1))
		return P(t, n, arr) * pi_tau_m(m, n, t-1)

	result = 0
	for s in range(t, n-m+j+1):
		result += P(t, s, arr) * Q(m, j+1, s+1, arr) * pi_tau(t-1, s)
	return result

def cp_likelihood(m, arr):
	n = len(arr)
	result = 0
	pbar = tqdm(range(1, n-m+1), desc='Training', unit='batch')
	for s in range(1, n-m+1):
		curr = P(1, s, arr) * Q(m, 1, s+1, arr)
		result += curr
		# if np.log10(curr)-np.log10(result) < -10:
		# 	break
		pbar.set_postfix({'rel prob': f'{np.log10(curr)-np.log10(result):.5f}'})
		pbar.update(1)
	return result

def m_prior(x, mu):
	return poisson.pmf(x, mu)

def pr_pos(j, t, t_last, arr, m):
	result = P(t_last+1, t, arr, True) + Q(m, j, t+1, arr, True) + np.log(pi_tau(t_last, t)) - Q(m, j-1, t_last+1, arr, True)
	print(np.log10(P(t_last+1, t, arr)), np.log10(Q(m, j, t+1, arr)), np.log10(pi_tau(t_last, t)), np.log10(Q(m, j-1, t_last+1, arr)))
	return result



