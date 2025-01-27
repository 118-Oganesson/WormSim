import wormsim_rs
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import base64
import io


class Worm:
    def __init__(self, gene, const, c_mode, concentration_map, using_rust=True):
        self.gene = gene
        self.c_mode = c_mode["c_mode"]

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

        if not using_rust:
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

        # 濃度マップの設定
        self.concentration_x_range = concentration_map["concentration_x_range"]
        self.concentration_y_range = concentration_map["concentration_y_range"]
        self.concentration_num = concentration_map["concentration_num"]
        self.color_scheme = concentration_map["color_scheme_blue"]
        self.opacity = concentration_map["opacity"]

    def _generate_weights(self):
        def gene_range(gene, min, max, num_values=1):
            return (
                [(gene + 1) / 2 * (max - min) + min] * num_values
                if num_values > 1
                else (gene + 1) / 2 * (max - min) + min
            )

        # 感覚ニューロン時間 [0.1, 4.2]
        N = gene_range(self.gene["gene"][0], 0.1, 4.2)
        M = gene_range(self.gene["gene"][1], 0.1, 4.2)

        # 介在ニューロンと運動ニューロンの閾値 [-15, 15]
        theta = np.zeros(8)
        theta[0] = gene_range(self.gene["gene"][2], -15, 15)
        theta[1] = gene_range(self.gene["gene"][3], -15, 15)
        theta[2] = gene_range(self.gene["gene"][4], -15, 15)
        theta[3] = gene_range(self.gene["gene"][5], -15, 15)
        theta[4], theta[5] = gene_range(self.gene["gene"][6], -15, 15, 2)
        theta[6], theta[7] = gene_range(self.gene["gene"][7], -15, 15, 2)

        # 感覚ニューロンON/OFFの重み [-15, 15]
        w_on = np.zeros(8)
        w_on[0] = gene_range(self.gene["gene"][8], -15, 15)
        w_on[1] = gene_range(self.gene["gene"][9], -15, 15)

        w_off = np.zeros(8)
        w_off[0] = gene_range(self.gene["gene"][10], -15, 15)
        w_off[1] = gene_range(self.gene["gene"][11], -15, 15)

        # 介在ニューロンと運動ニューロンのシナプス結合の重み [-15, 15]
        w = np.zeros((8, 8))
        w[0, 2] = gene_range(self.gene["gene"][12], -15, 15)
        w[1, 3] = gene_range(self.gene["gene"][13], -15, 15)
        w[2, 4], w[2, 5] = gene_range(self.gene["gene"][14], -15, 15, 2)
        w[3, 6], w[3, 7] = gene_range(self.gene["gene"][15], -15, 15, 2)
        w[4, 4], w[5, 5] = gene_range(self.gene["gene"][16], -15, 15, 2)
        w[6, 6], w[7, 7] = gene_range(self.gene["gene"][17], -15, 15, 2)

        # 介在ニューロンと運動ニューロンのギャップ結合の重み [0, 2.5]
        g = np.zeros((8, 8))
        g[0, 1], g[1, 0] = gene_range(self.gene["gene"][18], 0, 2.5, 2)
        g[2, 3], g[3, 2] = gene_range(self.gene["gene"][19], 0, 2.5, 2)

        # 運動ニューロンに入る振動成分の重み [0, 15]
        w_osc = np.zeros(8)
        w_osc[4], w_osc[7] = gene_range(self.gene["gene"][20], 0, 15, 2)
        w_osc[5], w_osc[6] = -w_osc[4], -w_osc[4]

        # 回転角度の重み [1, 3]
        w_nmj = gene_range(self.gene["gene"][21], 1, 3)

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

    def _c_(self, x_, y_):
        return self.alpha * np.sqrt((x_ - self.x_peak) ** 2 + (y_ - self.y_peak) ** 2)

    def _c_gauss(self, x_, y_):
        return self.c_0 * np.exp(
            -((x_ - self.x_peak) ** 2 + (y_ - self.y_peak) ** 2) / (2 * self.lambda_**2)
        )

    def _c_two_gauss(self, x_, y_):
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

    def _y_on_off(self, c_t):
        y_on = (
            np.sum(c_t[self.M_ : self.M_ + self.N_]) / self.N
            - np.sum(c_t[0 : self.M_]) / self.M
        )
        if y_on < 0:
            return 0, -y_on * 100 * self.dt
        else:
            return y_on * 100 * self.dt, 0

    def _sigmoid(self, x):
        return np.exp(np.minimum(x, 0)) / (1 + np.exp(-np.abs(x)))

    def _y_osc(self, t):
        return np.sin(2 * np.pi * t / self.T)

    def klinotaxis(self):
        # 濃度関数の選択
        concentration = {
            0: self._c_,
            1: self._c_gauss,
            2: self._c_two_gauss,
        }.get(self.c_mode, self._c_gauss)

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
            synapse = np.dot(self.w.T, self._sigmoid(y[:, k] + self.theta))
            gap = np.array(
                [np.dot(self.g[:, i], (y[:, k] - y[i, k])) for i in range(8)]
            )

            # 濃度の更新
            c_t = np.delete(c_t, 0)
            c_t = np.append(c_t, concentration(r[0, k], r[1, k]))

            # 介在ニューロンおよび運動ニューロンの膜電位の更新
            y_on, y_off = self._y_on_off(c_t)
            y[:, k + 1] = (
                y[:, k]
                + (
                    -y[:, k]
                    + synapse
                    + gap
                    + self.w_on * y_on
                    + self.w_off * y_off
                    + self.w_osc * self._y_osc(t[k])
                )
                / self.tau
                * self.dt
            )

            # 方向の更新
            phi[k] = self.w_nmj * (
                self._sigmoid(y[5, k] + self.theta[5])
                + self._sigmoid(y[6, k] + self.theta[6])
                - self._sigmoid(y[4, k] + self.theta[4])
                - self._sigmoid(y[7, k] + self.theta[7])
            )
            mu[k + 1] = mu[k] + phi[k] * self.dt

            # 位置の更新
            r[0, k + 1], r[1, k + 1] = (
                r[0, k] + self.v * np.cos(mu[k]) * self.dt,
                r[1, k] + self.v * np.sin(mu[k]) * self.dt,
            )

        return r

    def klinotaxis_rs(self):
        const = {
            "alpha": self.alpha,
            "c_0": self.c_0,
            "lambda": self.lambda_,
            "x_peak": self.x_peak,
            "y_peak": self.y_peak,
            "dt": self.dt,
            "periodic_time": self.T,
            "frequency": self.f,
            "mu_0": self.mu_0,
            "velocity": self.v,
            "simulation_time": self.time,
            "time_constant": self.tau,
        }

        return wormsim_rs.klinotaxis(self.gene, const, self.c_mode)

    def _generate_concentration_map(self, x, y):
        x, y = np.meshgrid(x, y)
        concentration = {
            0: self._c_,
            1: self._c_gauss,
            2: self._c_two_gauss,
        }.get(self.c_mode, self._c_gauss)
        return concentration(x, y)

    def _save_concentration_map_as_base64(self):
        """濃度マップを画像として保存し、Base64形式で返す"""
        x = np.linspace(
            self.concentration_x_range[0],
            self.concentration_x_range[1],
            num=self.concentration_num,
        )
        y = np.linspace(
            self.concentration_y_range[0],
            self.concentration_y_range[1],
            num=self.concentration_num,
        )
        z = self._generate_concentration_map(x, y)
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale=self.color_scheme,
                opacity=self.opacity,
                colorbar=dict(ticks="", tickvals=[], ticktext=[], len=0),
                showscale=False,
            )
        )
        fig.update_layout(
            width=self.concentration_num,
            height=self.concentration_num,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False, showticklabels=False, zeroline=False),
            yaxis=dict(visible=False, showticklabels=False, zeroline=False),
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
        )
        img_buffer = io.BytesIO()
        fig.write_image(img_buffer, format="PNG")
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode("utf-8")
        img_buffer.close()
        return img_base64

    def _image_to_base64(self, image):
        """画像をBase64形式に変換する関数"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        buffer.close()
        return img_base64

    def create_klintaxis_animation(
        self,
        trajectory,
        downsampling_factor=100,
        padding=1,
        animation_duration=10,
    ):
        # スタートポイントとピークポイントの定義
        start_point = [0, 0]
        peak_point = [self.x_peak, self.y_peak]

        # トラジェクトリーのダウンサンプリング
        x_downsampled, y_downsampled = (
            trajectory[0][::downsampling_factor],
            trajectory[1][::downsampling_factor],
        )

        # x_downsampled, y_downsampled, peak_point を使用して範囲を計算
        x_min = min(min(x_downsampled), peak_point[0])
        x_max = max(max(x_downsampled), peak_point[0])

        y_min = min(min(y_downsampled), peak_point[1])
        y_max = max(max(y_downsampled), peak_point[1])

        # 時間の計算とダウンサンプリング
        time = np.arange(0, self.time, self.dt)
        time_downsampled = time[::downsampling_factor]

        # 濃度マップ画像のBase64変換
        base64_concentration_image = self._save_concentration_map_as_base64()
        base64_concentration_source = (
            f"data:image/png;base64,{base64_concentration_image}"
        )

        # 線虫画像の処理（回転、反転）
        base_image = Image.open("./image/c_elegans.png")
        worm_rotated_image = base_image.rotate(-60, expand=True)
        worm_flipped_image = worm_rotated_image.transpose(Image.FLIP_LEFT_RIGHT)
        base64_worm_rotated_image = self._image_to_base64(worm_rotated_image)
        base64_worm_flipped_image = self._image_to_base64(worm_flipped_image)

        # プロットの作成
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=[start_point[0], 0.001],
                y=[start_point[1], 0.001],
                mode="lines",
                marker=dict(size=1, color="gray"),
                name="Trajectory",
            )
        )

        # 背景画像設定
        fig.add_layout_image(
            dict(
                source=base64_concentration_source,
                xref="x",
                yref="y",
                x=self.concentration_x_range[0],
                y=self.concentration_y_range[1],
                sizex=self.concentration_x_range[1] - self.concentration_x_range[0],
                sizey=self.concentration_y_range[1] - self.concentration_y_range[0],
                opacity=1,
                layer="below",
            )
        )

        # 開始点とピーク点のプロット
        fig.add_trace(
            go.Scatter(
                x=[start_point[0]],
                y=[start_point[1]],
                mode="markers",
                marker=dict(size=10, color="black"),
                name="Starting Point",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[peak_point[0]],
                y=[peak_point[1]],
                mode="markers",
                marker=dict(size=10, color="black", symbol="x"),
                name="Gradient Peak",
            )
        )

        if self.c_mode == 2:
            fig.add_trace(
                go.Scatter(
                    x=[-peak_point[0]],
                    y=[-peak_point[1]],
                    mode="markers",
                    marker=dict(size=10, color="black", symbol="x"),
                    name="Gradient Peak",
                )
            )

        # 1cmラインとテキスト
        fig.add_trace(
            go.Scatter(
                x=[peak_point[0] - 0.5, peak_point[0] + 0.5],
                y=[peak_point[1] - 1, peak_point[1] - 1],
                mode="lines",
                line=dict(color="black", width=1.5),
                name="Scale Line",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[peak_point[0]],
                y=[peak_point[1] - 0.95],
                mode="text",
                text=["1 cm"],
                textposition="top center",
                showlegend=False,
                textfont=dict(color="black"),
            )
        )

        # アニメーションフレーム作成
        frames = []
        for idx in range(1, len(x_downsampled)):
            dx = x_downsampled[idx] - x_downsampled[idx - 1]

            worm_image_source = "data:image/png;base64," + (
                base64_worm_rotated_image if dx > 0 else base64_worm_flipped_image
            )

            frames.append(
                go.Frame(
                    data=[
                        go.Scatter(
                            x=x_downsampled[:idx],
                            y=y_downsampled[:idx],
                            mode="lines",
                            line=dict(color="gray", width=2, shape="spline"),
                            name="Trajectory",
                        ),
                    ],
                    layout=go.Layout(
                        images=[
                            dict(
                                source=base64_concentration_source,
                                xref="x",
                                yref="y",
                                x=self.concentration_x_range[0],
                                y=self.concentration_y_range[1],
                                sizex=self.concentration_x_range[1]
                                - self.concentration_x_range[0],
                                sizey=self.concentration_y_range[1]
                                - self.concentration_y_range[0],
                                opacity=1,
                                layer="below",
                            ),
                            dict(
                                source=worm_image_source,
                                x=x_downsampled[idx],
                                y=y_downsampled[idx],
                                xref="x",
                                yref="y",
                                sizex=0.5,
                                sizey=0.5,
                                xanchor="center",
                                yanchor="middle",
                                layer="above",
                            ),
                        ]
                    ),
                    name=str(idx),
                )
            )

        # アニメーション設定
        fig.frames = frames
        fig.update_layout(
            # title="Salt Concentration Memory-Dependent Chemotaxis",
            # title_x=0.5,
            # title_xanchor="center",
            template="plotly_white",
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[x_min - padding, x_max + padding],
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[y_min - padding, y_max + padding],
            ),
            plot_bgcolor="white",
            xaxis_scaleanchor="y",
            legend=dict(
                x=0.5,
                y=0.0,
                xanchor="center",
                yanchor="top",
                orientation="h",
            ),
            dragmode="pan",
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {
                                        "duration": animation_duration,
                                        "redraw": True,
                                    },
                                    "fromcurrent": True,
                                },
                            ],
                            "label": "&#9654;",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "&#9724;",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.0,
                    "xanchor": "left",
                    "y": 0.11,
                    "yanchor": "top",
                    "font": {"color": "black"},
                }
            ],
            sliders=[
                {
                    "yanchor": "top",
                    "xanchor": "right",
                    "x": 1.0,
                    "y": 0.0,
                    "len": 0.8,
                    "steps": [
                        {
                            "args": [
                                [frame.name],
                                {
                                    "frame": {
                                        "duration": animation_duration,
                                        "redraw": True,
                                    },
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": f"{time_downsampled[idx]:.1f}s",
                            "method": "animate",
                        }
                        for idx, frame in enumerate(fig.frames)
                    ],
                    "font": {"color": "black"},
                }
            ],
        )

        return fig

    def create_concentration_map(
        self,
        padding=1,
    ):
        # スタートポイントとピークポイントの定義
        start_point = [0, 0]
        peak_point = [self.x_peak, self.y_peak]

        # 矢印を描画するためのベクトルを計算
        arrow_length = 1
        arrow_x = start_point[0] + arrow_length * np.cos(self.mu_0)
        arrow_y = start_point[1] + arrow_length * np.sin(self.mu_0)

        # 表示範囲を計算
        if self.c_mode == 1:
            x_min = min(start_point[0], peak_point[0])
            x_max = max(start_point[0], peak_point[0])

            y_min = min(start_point[1], peak_point[1])
            y_max = max(start_point[1], peak_point[1])

        elif self.c_mode == 2:
            x_min = min(-peak_point[0], peak_point[0])
            x_max = max(-peak_point[0], peak_point[0])

            y_min = min(-peak_point[1], peak_point[1])
            y_max = max(-peak_point[1], peak_point[1])

        x = np.linspace(
            self.concentration_x_range[0],
            self.concentration_x_range[1],
            num=self.concentration_num,
        )
        y = np.linspace(
            self.concentration_y_range[0],
            self.concentration_y_range[1],
            num=self.concentration_num,
        )
        z = self._generate_concentration_map(x, y)

        # 濃度のヒートマップのプロット
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale=self.color_scheme,
                opacity=self.opacity,
                colorbar=dict(
                    title="Concentration (mM)",
                    titleside="right",
                    tickformat=".2f",
                    len=1.0,
                    tickfont=dict(color="black"),
                    titlefont=dict(color="black"),
                ),
                showscale=True,
            )
        )

        # 開始点とピーク点のプロット
        fig.add_trace(
            go.Scatter(
                x=[start_point[0]],
                y=[start_point[1]],
                mode="markers",
                marker=dict(size=10, color="black"),
                name="Starting Point",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[peak_point[0]],
                y=[peak_point[1]],
                mode="markers",
                marker=dict(size=10, color="black", symbol="x"),
                name="Gradient Peak",
            )
        )

        if self.c_mode == 2:
            fig.add_trace(
                go.Scatter(
                    x=[-peak_point[0]],
                    y=[-peak_point[1]],
                    mode="markers",
                    marker=dict(size=10, color="black", symbol="x"),
                    name="Gradient Peak",
                )
            )

        # 1cmラインとテキスト
        fig.add_trace(
            go.Scatter(
                x=[peak_point[0] - 0.5, peak_point[0] + 0.5],
                y=[peak_point[1] - 1, peak_point[1] - 1],
                mode="lines",
                line=dict(color="black", width=1.5),
                name="Scale Line",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[peak_point[0]],
                y=[peak_point[1] - 0.95],
                mode="text",
                text=["1 cm"],
                textposition="top center",
                showlegend=False,
                textfont=dict(color="black"),
            )
        )

        fig.update_layout(
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[x_min - padding, x_max + padding],
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[y_min - padding, y_max + padding],
            ),
            plot_bgcolor="white",
            xaxis_scaleanchor="y",
            legend=dict(
                x=0.5,
                y=0.0,
                xanchor="center",
                yanchor="top",
                orientation="h",
            ),
            dragmode="pan",
            annotations=[
                go.layout.Annotation(
                    dict(
                        x=arrow_x,
                        y=arrow_y,
                        showarrow=True,
                        xref="x",
                        yref="y",
                        arrowcolor="black",
                        arrowsize=2,
                        arrowwidth=2,
                        ax=start_point[0],
                        ay=start_point[1],
                        axref="x",
                        ayref="y",
                        arrowhead=3,
                    )
                )
            ],
        )
        return fig
