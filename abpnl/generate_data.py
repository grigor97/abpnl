import numpy as np


def gen_directed_erdos_reyni_graph(d):
    edge_prob = 2 / (d - 1)

    A = np.random.binomial(1, edge_prob, size=d ** 2).reshape((d, d))
    A = np.triu(m=A, k=1)

    return A


def simulate_mult_pnl_erdos_renyi(n, d, name_noise="gaussian", name_h="cube"):
    h_inv = None
    sample_noise = None
    if name_noise == "gaussian" and name_h == "cube":
        def h_inv(x, pw=1 / 3):
            return np.sign(x) * abs(x) ** pw

        def sample_noise(t):
            return np.random.normal(size=t)

    elif name_noise == "evd" and name_h == "cube":
        def h_inv(x, pw=1 / 3):
            return np.sign(x) * abs(x) ** pw

        def sample_noise(t):
            return np.random.gumbel(size=t)

    def g(x):
        beta1 = np.random.uniform(-10, 10, x.shape[1])
        beta2 = np.random.uniform(-10, 10, x.shape[1])
        res = np.dot(x, beta1) + np.dot(x ** 2, beta2)

        return res

    A = gen_directed_erdos_reyni_graph(d)

    X = np.zeros((n, d))
    print(f"X shape {X.shape}")
    for j in range(d):
        parents = np.where(A[:, j] != 0)[0]
        if len(parents) == 0:
            X[:, j] = sample_noise(n)
        else:
            noise = sample_noise(n)
            z = g(X[:, parents]) + noise
            y = h_inv(z)
            X[:, j] = y

    return A, X.astype(np.float32)

