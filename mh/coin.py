from math import gamma, log, exp
import numpy as np
from collections import namedtuple
import scipy

APRIORI_T = 300
APRIORI_WEIGHT_STD = 100

PRIOR_K_MU = 0.1
PRIOR_LAMBDA_MU = 0.1
PRIOR_K_BETA = 0.1
PRIOR_LAMBDA_BETA = 0.1


def vectorize(f):
    return np.vectorize(f)


def calc_coin_num(m):
    ret = np.zeros(6)
    ret[5] = m // 500
    m -= ret[5] * 500
    ret[4] = m // 100
    m -= ret[4] * 100
    ret[3] = m // 50
    m -= ret[3] * 50
    ret[2] = m // 10
    m -= ret[2] * 10
    ret[1] = m // 5
    m -= ret[1] * 5
    ret[0] = m
    return ret


@vectorize
def calc_weight(m):
    coin_weight = [1, 3.75, 4.5, 4, 4.8, 7]
    coin_num = calc_coin_num(m)
    return np.sum(coin_weight * coin_num)


def llgamma(x, k, lmd):
    return k * log(lmd) - log(gamma(k)) + (k-1) * log(x) - lmd * x


def llpoisson(k, lmd):
    return k * log(lmd) - lmd - np.sum(np.log(np.arange(k) + 1))


def make_loglikelihood(Wobs):
    def loglikelihood(mu, T, M, beta):
        W = np.sum(calc_weight(M))
        k = W * beta
        lmd = beta
        l1 = llgamma(Wobs, k, lmd)     # Wobs
        l2 = - T * log(999)            # M
        l3 = llpoisson(T, mu)          # T
        l4 = llgamma(mu, APRIORI_EST_T, 0.01)    # mu
        l5 = llgamma(beta, 100, 0.01)  # beta
        return l1 + l2 + l3 + l4 + l5
    return loglikelihood


Step = namedtuple("Step", "mu T M beta loglik logsamp")


def make_sampler(Wobs):
    llfunc = make_loglikelihood(Wobs)

    def lsfunc(mu, T, M, beta):
        l1 = scipy.stats.gamma.logpdf(mu, APRIORI_EST_T, 1)
        l2 = scipy.stats.poisson.logpmf(T)
        l3 = -T * log(999)
        l4 = scipy.stats.gamma.logpdf(beta, APRIORI_BETA, 1)
        return 11 + l2 + l3 + l4

    def ret():
        mu = np.random.normal(APRIORI_EST_T, 1, 1)
        T = np.random.poisson(mu)
        M = np.random.randint(1, 1000, T)
        return Step(
            mu, T, M, llfunc(mu, T, M), lsfunc(mu, T, M)
        )
    return ret


def run(num_sample, Wobs):
    sampler = make_sampler(Wobs)
    current = sampler()
    samples = [current]
    for i in range(num_sample):
        next = sampler()
        acc_prob = min(1, )
        if np.random.uniform(0, 1) < acc_prob:
            samples.append(next)
        else:
            samples.append(current)
    return samples
