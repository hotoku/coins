from math import gamma, log, exp, lgamma
import numpy as np
from collections import namedtuple
import scipy
import logging
import time


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
    logging.debug(f"lmd={lmd}\tk={k}")
    return k * log(lmd) - lgamma(k) + (k-1) * log(x) - lmd * x


def llpoisson(k, lmd):
    return k * log(lmd) - lmd - np.sum(np.log(np.arange(k) + 1))


class Sampler:
    def __init__(self, observation, hyper_parameter):
        raise NotImplemented(
            f"{type(self)}: __init__ is not implemented")

    def log_likelihood(self, prm):
        raise NotImplemented(
            f"{type(self)}: log_likelihood is not implemented")

    def log_sampling_distribution(self, new, current):
        raise NotImplemented(
            f"{type(self)}: log_sampling_distribution is not implemented")

    def sample(prm):
        raise NotImplemented(
            f"{type(self)}: sample is not implemented")


class Coin(Sampler):
    Param = namedtuple("Param", "mu, T, M, beta")

    def __init__(self, observation, hyper_parameter):
        self.Wobs = observation
        self.mb, self.sb, self.mmu, self.smu, self.sb_, self.smu_ = hyper_parameter

    def log_likelihood(self, prm):
        mu, T, M, beta = prm
        lb, kb = self.mb / self.sb**2, self.mb**2 / self.sb**2
        lmu, kmu = self.mmu / self.smu**2, self.mmu**2 / self.smu**2
        W = np.sum(calc_weight(M))
        lmd, k = W / beta**2, W**2 / beta**2
        l1 = llgamma(self.Wobs, k, lmd)
        l2 = - T * log(999)
        l3 = llpoisson(T, mu)
        l4 = llgamma(mu, kmu, lmu)
        l5 = llgamma(beta, kb, lb)
        return l1 + l2 + l3 + l4 + l5

    def log_sampling_distribution(self, new, current):
        mu_, T_, M_, beta_ = new
        mu, T, M, beta = current
        lb, kb = beta / self.sb_**2, beta**2 / self.sb_**2
        lmu, kmu = mu / self.smu_**2, mu**2 / self.smu_**2
        l1 = -T_ * log(999)
        l2 = llpoisson(T_, T)
        l3 = llgamma(mu_, kmu, lmu)
        l4 = llgamma(beta_, kb, lb)
        return 11 + l2 + l3 + l4

    def sample(self, prm):
        mu, T, M, beta = prm
        lb, kb = beta / self.sb_**2, beta**2 / self.sb_**2
        lmu, kmu = mu / self.smu_**2, mu**2 / self.smu_**2
        mu_ = np.random.gamma(kmu, lmu)
        T_ = np.random.poisson(T)
        M_ = np.random.randint(1, 1000, T_)
        beta_ = np.random.gamma(kb, lb)
        return Coin.Param(mu_, T_, M_, beta_)


def run(num_sample, observation,
        hyper_parameter,
        initial_values,
        sampler_class,
        seed=0):
    np.random.seed(seed)
    sampler = sampler_class(observation, hyper_parameter)
    current = initial_values
    samples = []
    lls = []
    proposed = []
    logging.debug(f"initial value: {current}")
    start_time = time.time()
    for i in range(num_sample):
        samples.append(current)
        lls.append(sampler.log_likelihood(current))
        if (i+1) % 100 == 0:
            print(f"{i+1}-th step begins: {time.time() - start_time}")
        new = sampler.sample(current)
        proposed.append(new)
        logging.debug(f"proposed value: {new}")
        lps = np.array([
            sampler.log_likelihood(new),
            sampler.log_sampling_distribution(current, new),
            - sampler.log_likelihood(current),
            - sampler.log_sampling_distribution(new, current)
        ])
        logging.debug(f"lps={lps}")
        sum_lps = np.sum(lps)
        acc_prob = 1 if sum_lps >= 0 else exp(sum_lps)
        if np.random.uniform(0, 1) < acc_prob:
            samples.append(new)
            current = new
            logging.debug("proposal is accepted")
        else:
            samples.append(current)
            logging.debug("proposal is rejected")
    Ret = namedtuple("Ret", "samples, lls, proposed")
    return Ret(samples, lls, proposed)
