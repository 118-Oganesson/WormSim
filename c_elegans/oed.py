import numpy as np
# import load


def c_(alpha, x_, y_, x_peak, y_peak):
    return alpha * np.sqrt((x_ - x_peak) ** 2 + (y_ - y_peak) ** 2)


def c_gauss(c_0, lambda_, x_, y_, x_peak, y_peak):
    return c_0 * np.exp(-((x_ - x_peak) ** 2 + (y_ - y_peak) ** 2) / (2 * lambda_**2))


def c_two_gauss(c_0, lambda_, x_, y_, x_peak, y_peak):
    return c_0 * (
        np.exp(-((x_ - x_peak) ** 2 + (y_ - y_peak) ** 2) / (2 * lambda_**2))
        - np.exp(-((x_ + x_peak) ** 2 + (y_ + y_peak) ** 2) / (2 * lambda_**2))
    )


def y_on(c_t, N_, M_, N, M, dt):
    y_ = np.sum(c_t[M_ : M_ + N_]) / N - np.sum(c_t[0:M_]) / M
    y_ = np.clip(y_, 0, None)
    return y_ * 100 * dt


def y_off(c_t, N_, M_, N, M, dt):
    y_ = np.sum(c_t[0:M_]) / M - np.sum(c_t[M_ : M_ + N_]) / N
    y_ = np.clip(y_, 0, None)
    return y_ * 100 * dt


def sigmoid(x):
    return np.exp(np.minimum(x, 0)) / (1 + np.exp(-np.abs(x)))


def y_osc(t, T):
    return np.sin(2 * np.pi * t / T)


def gene_range_1(gene, min, max):
    gene_ = (gene + 1) / 2 * (max - min) + min
    return gene_


def gene_range_2(gene, min, max):
    gene_ = (gene + 1) / 2 * (max - min) + min
    return gene_, gene_


def weight(gene):
    # 遺伝子の引き渡し
    # 感覚ニューロン時間 [0.1,4.2]
    N = gene_range_1(gene[0], 0.1, 4.2)
    M = gene_range_1(gene[1], 0.1, 4.2)

    # 介在ニューロンと運動ニューロンの閾値 [-15.15]
    theta = np.zeros(8)
    theta[0] = gene_range_1(gene[2], -15, 15)
    theta[1] = gene_range_1(gene[3], -15, 15)
    theta[2] = gene_range_1(gene[4], -15, 15)
    theta[3] = gene_range_1(gene[5], -15, 15)
    theta[4], theta[5] = gene_range_2(gene[6], -15, 15)
    theta[6], theta[7] = gene_range_2(gene[7], -15, 15)

    # 感覚ニューロンONの重み [-15.15]
    w_on = np.zeros(8)
    w_on[0] = gene_range_1(gene[8], -15, 15)
    w_on[1] = gene_range_1(gene[9], -15, 15)

    # 感覚ニューロンOFFの重み [-15.15]
    w_off = np.zeros(8)
    w_off[0] = gene_range_1(gene[10], -15, 15)
    w_off[1] = gene_range_1(gene[11], -15, 15)

    # 介在ニューロンと運動ニューロンのシナプス結合の重み [-15.15]
    w = np.zeros((8, 8))
    w[0, 2] = gene_range_1(gene[12], -15, 15)
    w[1, 3] = gene_range_1(gene[13], -15, 15)
    w[2, 4], w[2, 5] = gene_range_2(gene[14], -15, 15)
    w[3, 6], w[3, 7] = gene_range_2(gene[15], -15, 15)
    w[4, 4], w[5, 5] = gene_range_2(gene[16], -15, 15)
    w[6, 6], w[7, 7] = gene_range_2(gene[17], -15, 15)

    # 介在ニューロンと運動ニューロンのギャップ結合の重み [0.2.5]
    g = np.zeros((8, 8))
    g[0, 1], g[1, 0] = gene_range_2(gene[18], 0, 2.5)
    g[2, 3], g[3, 2] = gene_range_2(gene[19], 0, 2.5)

    # 運動ニューロンに入る振動成分の重み [0.15]
    w_osc = np.zeros(8)
    w_osc[4], w_osc[7] = gene_range_2(gene[20], 0, 15)
    w_osc[5], w_osc[6] = -w_osc[4], -w_osc[4]

    # 回転角度の重み [1,3]
    w_nmj = gene_range_1(gene[21], 1, 3)

    return N, M, theta, w_on, w_off, w, g, w_osc, w_nmj


def constant(const):
    alpha = const["alpha"]
    x_peak = const["x_peak"]
    y_peak = const["y_peak"]
    dt = const["dt"]
    T = const["periodic_time"]
    f = const["frequency"]
    v = const["velocity"]
    time = const["simulation_time"]
    tau = const["time_constant"]
    c_0 = const["c_0"]
    lambda_ = const["lambda"]

    return alpha, x_peak, y_peak, dt, T, f, v, time, tau, c_0, lambda_


def time_constant_step(gene, const):
    N, M, theta, w_on, w_off, w, g, w_osc, w_nmj = weight(gene)
    alpha, x_peak, y_peak, dt, T, f, v, time, tau, c_0, lambda_ = constant(const)
    # 時間に関する定数をステップ数に変換
    N_ = np.floor(N / dt).astype(int)
    M_ = np.floor(M / dt).astype(int)
    f_inv = np.floor(1 / f / dt).astype(int)
    T_ = np.floor(T / dt).astype(int)

    return N_, M_, f_inv, T_


def klinotaxis(gene, const, c_mode):
    # 遺伝子の値をスケーリング
    N, M, theta, w_on, w_off, w, g, w_osc, w_nmj = weight(gene)

    # tomlファイルの読み込み
    alpha, x_peak, y_peak, dt, T, f, v, time, tau, c_0, lambda_ = constant(const)

    # 時間に関する定数をステップ数に変換
    N_, M_, f_inv, T_ = time_constant_step(gene, const)

    # 各種配列の初期化
    t = np.arange(0, time, dt)
    c_t = np.zeros(N_ + M_)
    if c_mode == 0:
        c_t[0 : N_ + M_] = c_(alpha, 0, 0, x_peak, y_peak)
    elif c_mode == 1:
        c_t[0 : N_ + M_] = c_gauss(c_0, lambda_, 0, 0, x_peak, y_peak)
    elif c_mode == 2:
        c_t[0 : N_ + M_] = c_two_gauss(c_0, lambda_, 0, 0, x_peak, y_peak)
    y = np.zeros((8, len(t)))
    y[4:8, 0] = np.random.rand(4)  # 運動ニューロンの活性を0～1の範囲でランダム化
    phi = np.zeros(len(t))
    mu = np.zeros(len(t))
    mu[0] = const["mu_0"]
    r = np.zeros((2, len(t)))

    # オイラー法
    for k in range(len(t) - 1):
        # シナプス結合およびギャップ結合からの入力
        synapse = np.dot(w.T, sigmoid(y[:, k] + theta))
        gap = np.array([np.dot(g[:, i], (y[:, k] - y[i, k])) for i in range(8)])

        # 濃度の更新
        c_t = np.delete(c_t, 0)
        if c_mode == 0:
            c = c_(alpha, r[0, k], r[1, k], x_peak, y_peak)
        elif c_mode == 1:
            c = c_gauss(c_0, lambda_, r[0, k], r[1, k], x_peak, y_peak)
        elif c_mode == 2:
            c = c_two_gauss(c_0, lambda_, r[0, k], r[1, k], x_peak, y_peak)
        c_t = np.append(c_t, c)

        # 介在ニューロンおよび運動ニューロンの膜電位の更新
        y[:, k + 1] = (
            y[:, k]
            + (
                -y[:, k]
                + synapse
                + gap
                + w_on * y_on(c_t, N_, M_, N, M, dt)
                + w_off * y_off(c_t, N_, M_, N, M, dt)
                + w_osc * y_osc(t[k], T)
            )
            / tau
            * dt
        )

        # 方向の更新
        phi[k] = w_nmj * (
            sigmoid(y[5, k] + theta[5])
            + sigmoid(y[6, k] + theta[6])
            - sigmoid(y[4, k] + theta[4])
            - sigmoid(y[7, k] + theta[7])
        )
        mu[k + 1] = mu[k] + phi[k] * dt

        # 位置の更新
        r[0, k + 1], r[1, k + 1] = (
            r[0, k] + v * np.cos(mu[k]) * dt,
            r[1, k] + v * np.sin(mu[k]) * dt,
        )

    return r
