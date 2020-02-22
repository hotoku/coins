n_iter = int(1e4)

params = [0, 0, 0, 0]


def shoud_accept(p, np):
    pass


def sample_from_proposal(params):
    pass


def likelihood():
    pass


for i in range(n_iter):
    new_params = sample_from_proposal(params)
    if shoud_accept(params, new_params):
        params = new_params
