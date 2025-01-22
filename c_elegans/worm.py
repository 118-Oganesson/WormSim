import numpy as np


class Worm:
    def __init__(self, gene, const, c_mode):
        self.gene = gene
        self.const = const
        self.c_mode = c_mode

        # 定数の設定
        self.alpha = const["alpha"]
        self.c_0 = const["c_0"]
        self.lambda_ = const["lambda"]
        self.x_peak = const["x_peak"]
        self.y_peak = const["y_peak"]
        self.dt = const["dt"]
        self.T = const["periodic_time"]
        self.f = const["frequency"]
        self.mu_0 = const["mu_0"]
        self.v = const["velocity"]
        self.time = const["simulation_time"]
        self.tau = const["time_constant"]

        # 遺伝子パラメータの設定
        weights = self._generate_weights()
        self.N = weights["N"]
        self.M = weights["M"]
        self.theta = weights["theta"]
        self.w_on = weights["w_on"]
        self.w_off = weights["w_off"]
        self.w = weights["w"]
        self.g = weights["g"]
        self.w_osc = weights["w_osc"]
        self.w_nmj = weights["w_nmj"]

        # 時間に関する定数の設定
        params = self._generate_time_constant_to_step()
        self.N_ = params["N_"]
        self.M_ = params["M_"]
        self.f_inv = params["f_inv"]
        self.T_ = params["T_"]

    def _generate_weights(self):
        def gene_range(gene, min, max, num_values=1):
            return (
                [(gene + 1) / 2 * (max - min) + min] * num_values
                if num_values > 1
                else (gene + 1) / 2 * (max - min) + min
            )

        # 感覚ニューロン時間 [0.1, 4.2]
        N = gene_range(self.gene[0], 0.1, 4.2)
        M = gene_range(self.gene[1], 0.1, 4.2)

        # 介在ニューロンと運動ニューロンの閾値 [-15, 15]
        theta = np.zeros(8)
        theta[0] = gene_range(self.gene[2], -15, 15)
        theta[1] = gene_range(self.gene[3], -15, 15)
        theta[2] = gene_range(self.gene[4], -15, 15)
        theta[3] = gene_range(self.gene[5], -15, 15)
        theta[4], theta[5] = gene_range(self.gene[6], -15, 15, 2)
        theta[6], theta[7] = gene_range(self.gene[7], -15, 15, 2)

        # 感覚ニューロンON/OFFの重み [-15, 15]
        w_on = np.zeros(8)
        w_on[0] = gene_range(self.gene[8], -15, 15)
        w_on[1] = gene_range(self.gene[9], -15, 15)

        w_off = np.zeros(8)
        w_off[0] = gene_range(self.gene[10], -15, 15)
        w_off[1] = gene_range(self.gene[11], -15, 15)

        # 介在ニューロンと運動ニューロンのシナプス結合の重み [-15, 15]
        w = np.zeros((8, 8))
        w[0, 2] = gene_range(self.gene[12], -15, 15)
        w[1, 3] = gene_range(self.gene[13], -15, 15)
        w[2, 4], w[2, 5] = gene_range(self.gene[14], -15, 15, 2)
        w[3, 6], w[3, 7] = gene_range(self.gene[15], -15, 15, 2)
        w[4, 4], w[5, 5] = gene_range(self.gene[16], -15, 15, 2)
        w[6, 6], w[7, 7] = gene_range(self.gene[17], -15, 15, 2)

        # 介在ニューロンと運動ニューロンのギャップ結合の重み [0, 2.5]
        g = np.zeros((8, 8))
        g[0, 1], g[1, 0] = gene_range(self.gene[18], 0, 2.5, 2)
        g[2, 3], g[3, 2] = gene_range(self.gene[19], 0, 2.5, 2)

        # 運動ニューロンに入る振動成分の重み [0, 15]
        w_osc = np.zeros(8)
        w_osc[4], w_osc[7] = gene_range(self.gene[20], 0, 15, 2)
        w_osc[5], w_osc[6] = -w_osc[4], -w_osc[4]

        # 回転角度の重み [1, 3]
        w_nmj = gene_range(self.gene[21], 1, 3)

        return {
            "N": N,
            "M": M,
            "theta": theta,
            "w_on": w_on,
            "w_off": w_off,
            "w": w,
            "g": g,
            "w_osc": w_osc,
            "w_nmj": w_nmj,
        }

    def _generate_time_constant_to_step(self):
        # 時間に関する定数をステップ数に変換
        N_ = np.floor(self.N / self.dt).astype(int)
        M_ = np.floor(self.M / self.dt).astype(int)
        f_inv = np.floor(1 / self.f / self.dt).astype(int)
        T_ = np.floor(self.T / self.dt).astype(int)

        return {"N_": N_, "M_": M_, "f_inv": f_inv, "T_": T_}

    def c_(self, x_, y_):
        return self.alpha * np.sqrt((x_ - self.x_peak) ** 2 + (y_ - self.y_peak) ** 2)

    def c_gauss(self, x_, y_):
        return self.c_0 * np.exp(
            -((x_ - self.x_peak) ** 2 + (y_ - self.y_peak) ** 2) / (2 * self.lambda_**2)
        )

    def c_two_gauss(self, x_, y_):
        return self.c_0 * (
            np.exp(
                -((x_ - self.x_peak) ** 2 + (y_ - self.y_peak) ** 2)
                / (2 * self.lambda_**2)
            )
            - np.exp(
                -((x_ + self.x_peak) ** 2 + (y_ + self.y_peak) ** 2)
                / (2 * self.lambda_**2)
            )
        )

    def y_on_off(self, c_t):
        y_on = (
            np.sum(c_t[self.M_ : self.M_ + self.N_]) / self.N
            - np.sum(c_t[0 : self.M_]) / self.M
        )
        if y_on < 0:
            return 0, -y_on * 100 * self.dt
        else:
            return y_on * 100 * self.dt, 0

    def sigmoid(self, x):
        return np.exp(np.minimum(x, 0)) / (1 + np.exp(-np.abs(x)))

    def y_osc(self, t):
        return np.sin(2 * np.pi * t / self.T)

    def klinotaxis(
        self,
    ):
        # 濃度関数の選択
        concentration = {
            0: self.c_,
            1: self.c_gauss,
            2: self.c_two_gauss,
        }.get(self.c_mode, self.c_gauss)

        # 各種配列の初期化
        t = np.arange(0, self.time, self.dt)
        c_t = np.zeros(self.N_ + self.M_)
        c_t[0 : self.N_ + self.M_] = concentration(0, 0)
        y = np.zeros((8, len(t)))
        y[4:8, 0] = np.random.rand(4)  # 運動ニューロンの活性を0～1の範囲でランダム化
        phi = np.zeros(len(t))
        mu = np.zeros(len(t))
        mu[0] = self.mu_0
        r = np.zeros((2, len(t)))

        # オイラー法
        for k in range(len(t) - 1):
            # シナプス結合およびギャップ結合からの入力
            synapse = np.dot(self.w.T, self.sigmoid(y[:, k] + self.theta))
            gap = np.array([np.dot(self.g[:, i], (y[:, k] - y[i, k])) for i in range(8)])

            # 濃度の更新
            c_t = np.delete(c_t, 0)
            c_t = np.append(c_t, concentration(r[0, k], r[1, k]))

            # 介在ニューロンおよび運動ニューロンの膜電位の更新
            y_on, y_off = self.y_on_off(c_t)
            y[:, k + 1] = (
                y[:, k]
                + (
                    -y[:, k]
                    + synapse
                    + gap
                    + self.w_on * y_on
                    + self.w_off * y_off
                    + self.w_osc * self.y_osc(t[k])
                )
                / self.tau
                * self.dt
            )

            # 方向の更新
            phi[k] = self.w_nmj * (
                self.sigmoid(y[5, k] + self.theta[5])
                + self.sigmoid(y[6, k] + self.theta[6])
                - self.sigmoid(y[4, k] + self.theta[4])
                - self.sigmoid(y[7, k] + self.theta[7])
            )
            mu[k + 1] = mu[k] + phi[k] * self.dt

            # 位置の更新
            r[0, k + 1], r[1, k + 1] = (
                r[0, k] + self.v * np.cos(mu[k]) * self.dt,
                r[1, k] + self.v * np.sin(mu[k]) * self.dt,
            )

        return r
