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


class Normal(Sampler):
    """
    x ~ normal(0, 1)
    y ~ normal(0, 1)
    z ~ noraml(x + y, s1)
    """
    Param = namedtuple("Param", "x y")

    def __init__(self, observation, hyper_parameter):
        self.z = observation
        self.s1 = hyper_parameter

    def log_likelihood(self, prm):
        x, y = prm
        l1 = scipy.stats.norm.logpdf(x)
        l2 = scipy.stats.norm.logpdf(y)
        l3 = scipy.stats.norm.logpdf(self.z, x+y, self.s1)
        return l1 + l2 + l3

    def log_sampling_distribution(self, to_, from_):
        x_, y_ = to_
        x, y = from_
        l1 = scipy.stats.norm.logpdf(x_, x)
        l2 = scipy.stats.norm.logpdf(y_, y)
        return l1 + l2

    def sample(self, prm):
        x, y = prm
        x_ = scipy.stats.norm.rvs(loc=x, size=1)[0]
        y_ = scipy.stats.norm.rvs(loc=y, size=1)[0]
        return Normal.Param(x_, y_)


class Coin(Sampler):
    Param = namedtuple("Param", "mu T M beta")

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

    def log_sampling_distribution(self, to_, from_):
        """
        from_ → to_ に遷移する確率
        """
        mu_, T_, M_, beta_ = to_
        mu, T, M, beta = from_
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
        thmu, thb = 1/lmu, 1/lb
        mu_ = np.random.gamma(kmu, thmu)
        T_ = np.random.poisson(T)
        M_ = np.random.randint(1, 1000, T_)
        beta_ = np.random.gamma(kb, thb)
        return Coin.Param(mu_, T_, M_, beta_)


def l(e, v):
    return e / (v**2)


def k(e, v):
    return (e**2) / (v**2)


class Coin2(Sampler):
    Param = namedtuple("Param", "T M mu beta")

    def __init__(self, observation, hyper_parameter):
        self.Wo = observation
        self.m_beta, self.s_beta, self.m_mu, self.s_mu, self.s_beta_, self.s_mu_ = hyper_parameter
        self.k_mu = k(self.m_mu, self.s_mu)
        self.l_mu = l(self.m_mu, self.s_mu)
        self.k_beta = k(self.m_beta, self.s_beta)
        self.l_beta = l(self.m_beta, self.s_beta)

    def log_likelihood(self, prm):
        T, M, mu, beta = prm
        l1 = llgamma(mu, self.k_mu, self.l_mu)
        l2 = llgamma(beta, self.k_beta, self.l_beta)
        l3 = llpoisson(T, mu)
        l4 = -T * log(999)

        W = np.sum(calc_weight(M))
        k_Wo = k(W, beta)
        l_Wo = l(W, beta)
        l5 = llgamma(self.Wo, k_Wo, l_Wo)
        return l1 + l2 + l3 + l4 + l5

    def log_sampling_distribution(self, to_, from_):
        T_, M_, mu_, beta_ = to_
        T, M, mu, beta = from_

        kb = k(beta, self.s_beta_)
        lb = l(beta, self.s_beta_)
        l1 = llgamma(beta_, kb, lb)

        km = k(mu, self.s_mu_)
        lm = l(mu, self.s_mu_)
        l2 = llgamma(mu_, km, lm)

        l3 = -T_ * log(999)
        l4 = llpoisson(T_, T)

        return l1 + l2 + l3 + l4

    def sample(self, prm):
        T, M, mu, beta = prm
        kb = k(beta, self.s_beta_)
        lb = l(beta, self.s_beta_)
        beta_ = np.random.gamma(kb, 1/lb)

        km = k(mu, self.s_mu_)
        lm = l(mu, self.s_mu_)
        mu_ = np.random.gamma(km, 1/lm)

        T_ = np.random.poisson(T)
        M_ = np.random.randint(1, 1000, T_)

        return Coin2.Param(T_, M_, mu_, beta_)


class Simple(Sampler):
    Param = namedtuple("Param", "T M muT vW")

    def __init__(self, observation, hyper_parameter):
        self.W = observation
        self.e_muT, self.v_muT, self.e_vW, self.v_vW, self.v_muT_, self.v_vW_ = hyper_parameter
        self.k_muT, self.l_muT = k(
            self.e_muT, self.v_muT), l(self.e_muT, self.v_muT)
        self.k_vW, self.l_vW = k(self.e_vW, self.v_vW), l(self.e_vW, self.v_vW)

    def log_likelihood(self, prm):
        T, M, muT, vW = prm
        l1 = llgamma(vW, self.k_vW, self.l_vW)
        l2 = llgamma(muT, self.k_muT, self.l_muT)
        l3 = llpoisson(T, muT)
        l4 = -T * log(999)
        eW = np.sum(M)
        k_W, l_W = k(eW, vW), l(eW, vW)
        l5 = llgamma(self.W, k_W, l_W)
        return l1 + l2 + l3 + l4 + l5

    def log_sampling_distribution(self, to_, from_):
        T_, M_, muT_, vW_ = to_
        T, M, muT, vW = from_
        l1 = llpoisson(T_, T)
        l2 = -T_ * log(999)
        k_muT, l_muT = k(muT, self.v_muT_), l(muT, self.v_muT_)
        k_vW, l_vW = k(vW, self.v_vW_), l(vW, self.v_vW_)
        l3 = llgamma(muT_, k_muT, l_muT)
        l4 = llgamma(vW_, k_vW, l_vW)
        return l1 + l2 + l3 + l4

    def sample(self, prm):
        T, M, muT, vW = prm
        k_muT, l_muT = k(muT, self.v_muT_), l(muT, self.v_muT_)
        k_vW, l_vW = k(vW, self.v_vW_), l(vW, self.v_vW_)
        T_ = np.random.poisson(T)
        M_ = np.random.randint(1, 1000, T_)
        muT_ = np.random.gamma(k_muT, 1/l_muT)
        vW_ = np.random.gamma(k_vW, 1/l_vW)
        return Simple.Param(T_, M_, muT_, vW_)


class McmcException(Exception):
    def __init__(self, ret, e):
        self.ret = ret
        self.e = e


def run(num_sample, observation,
        hyper_parameter,
        initial_values,
        sampler_class,
        seed=0,
        print_step=1000):
    Ret = namedtuple("Ret", "samples, lls, proposed")

    np.random.seed(seed)
    sampler = sampler_class(observation, hyper_parameter)
    current = initial_values
    samples = []
    lls = []
    proposed = []
    logging.debug(f"initial value: {current}")
    start_time = time.time()
    try:
        for i in range(num_sample):
            samples.append(current)
            lls.append(sampler.log_likelihood(current))
            if (i+1) % print_step == 0:
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
                current = new
                logging.debug("proposal is accepted")
            else:
                logging.debug("proposal is rejected")
        return Ret(samples, lls, proposed)

    except Exception as e:
        raise McmcException(Ret(samples, lls, proposed), e)
