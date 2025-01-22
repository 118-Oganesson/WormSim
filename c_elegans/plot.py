import c_elegans.oed as oed
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import base64
import io


def downsample_data(x, y, factor):
    """ダウンサンプリング関数"""
    return x[::factor], y[::factor]


def save_concentration_map_as_base64(
    x, y, z, colormap="Viridis", opacity=1, size=(500, 500)
):
    """濃度マップを画像として保存し、Base64形式で返す"""
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=colormap,
            opacity=opacity,
            colorbar=dict(ticks="", tickvals=[], ticktext=[], len=0),
            showscale=False,
        )
    )
    fig.update_layout(
        width=size[0],
        height=size[1],
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


def image_to_base64(image):
    """画像をBase64形式に変換する関数"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    return img_base64


def compute_concentration_map(x, y, const, c_mode):
    """指定されたxおよびy範囲に対する濃度関数を計算します。"""
    alpha, x_peak, y_peak, dt, T, f, v, time, tau, c_0, lambda_ = oed.constant(const)
    x, y = np.meshgrid(x, y)
    if c_mode == 0:
        z = oed.c_(alpha, x, y, x_peak, y_peak)
    elif c_mode == 1:
        z = oed.c_gauss(c_0, lambda_, x, y, x_peak, y_peak)
    elif c_mode == 2:
        z = oed.c_two_gauss(c_0, lambda_, x, y, x_peak, y_peak)
    return z


def create_klintaxis_animation(
    x,
    y,
    const,
    c_mode,
    downsampling_factor=100,
    x_range=(-15, 15),
    y_range=(-15, 15),
    color_scheme=None,
    start_point=[0, 0],
    peak_point=[4.5, 0],
    font_size=12,
    max_y=1,
    time_factor=50,
    animation_duration=50,
):
    """
    Klinotaxisデータの可視化とアニメーション作成

    Args:
    - x, y: 移動経路データ
    - const: シミュレーション定数
    - compute_concentration_map: 濃度マップを計算する関数
    - downsampling_factor: ダウンサンプリングファクター
    - x_range, y_range: x, y軸の範囲
    - color_scheme: カラースキーム
    - font_size: フォントサイズ
    - max_y: 最大y座標
    - time_factor: アニメーションフレームの間隔
    - animation_duration: アニメーションのフレーム更新間隔

    Returns:
    - fig: PlotlyのFigureオブジェクト
    """
    # ダウンサンプリング
    x_downsampled, y_downsampled = downsample_data(x, y, downsampling_factor)

    # 濃度マップ計算
    x_vals = np.linspace(x_range[0], x_range[1], num=500)
    y_vals = np.linspace(y_range[0], y_range[1], num=500)
    z_vals = compute_concentration_map(x_vals, y_vals, const, c_mode)

    # カラーマップ設定
    if color_scheme is None:
        color_scheme = [
            "#ffffff",
            "#f7fbff",
            "#eff7ff",
            "#dfefff",
            "#cfe7ff",
            "#bfdfff",
            "#afd7ff",
            "#9fcfff",
            "#8fc7ff",
        ]

    # 濃度マップ画像のBase64変換
    base64_concentration_image = save_concentration_map_as_base64(
        x_vals, y_vals, z_vals, color_scheme
    )
    base64_concentration_source = f"data:image/png;base64,{base64_concentration_image}"

    # ベース画像の処理（反転、回転）
    base_image = Image.open("./c_elegans.png")
    rotated_image = base_image.rotate(-60, expand=True)
    flipped_image = rotated_image.transpose(Image.FLIP_LEFT_RIGHT)
    base64_rotated_image = image_to_base64(rotated_image)
    base64_flipped_image = image_to_base64(flipped_image)

    # プロットの作成
    fig = go.Figure()

    # 背景画像設定
    fig.add_layout_image(
        dict(
            source=base64_concentration_source,
            xref="x",
            yref="y",
            x=x_range[0],
            y=y_range[1],
            sizex=x_range[1] - x_range[0],
            sizey=y_range[1] - y_range[0],
            opacity=1,
            layer="below",
        )
    )

    # 開始点とピーク点のプロット
    fig.add_trace(
        go.Scatter(
            x=[start_point[0], peak_point[0]],
            y=[start_point[1], peak_point[1]],
            mode="markers",
            marker=dict(size=5, color="black"),
            showlegend=False,
        )
    )

    # 開始点とピーク点の縦線
    fig.add_trace(
        go.Scatter(
            x=[start_point[0], start_point[0]],
            y=[start_point[1], max_y],
            mode="lines",
            line=dict(color="black", dash="solid", width=0.5),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[peak_point[0], peak_point[0]],
            y=[peak_point[1], max_y],
            mode="lines",
            line=dict(color="black", dash="solid", width=0.5),
            showlegend=False,
        )
    )

    # 開始点とピーク点のラベル
    fig.add_trace(
        go.Scatter(
            x=[start_point[0]],
            y=[max_y + 0.1],
            mode="text",
            text=["Starting Point"],
            textposition="top center",
            textfont=dict(size=font_size),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[peak_point[0]],
            y=[max_y + 0.1],
            mode="text",
            text=["Gradient Peak"],
            textposition="top center",
            textfont=dict(size=font_size),
            showlegend=False,
        )
    )

    # 1cmラインとテキスト
    fig.add_trace(
        go.Scatter(
            x=[4, 5],
            y=[-1, -1],
            mode="lines",
            line=dict(color="black", width=1.5),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[4.5],
            y=[-0.95],
            mode="text",
            text=["1 cm"],
            textposition="top center",
            textfont=dict(size=font_size),
            showlegend=False,
        )
    )

    # アニメーションフレーム作成
    frames = []
    for idx in range(1, len(x_downsampled)):
        dx = x_downsampled[idx] - x_downsampled[idx - 1]

        image_source = "data:image/png;base64," + (
            base64_rotated_image if dx > 0 else base64_flipped_image
        )

        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=x_downsampled[:idx],
                        y=y_downsampled[:idx],
                        mode="lines",
                        line=dict(color="gray", width=2),
                    ),
                ],
                layout=go.Layout(
                    images=[
                        dict(
                            source=base64_concentration_source,
                            xref="x",
                            yref="y",
                            x=x_range[0],
                            y=y_range[1],
                            sizex=x_range[1] - x_range[0],
                            sizey=y_range[1] - y_range[0],
                            opacity=1,
                            layer="below",
                        ),
                        dict(
                            source=image_source,
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

    # 時間の計算とダウンサンプリング
    time = np.array([i * const["dt"] for i in range(len(x))])
    time_downsampled = time[::downsampling_factor]

    # アニメーション設定
    fig.frames = frames
    fig.update_layout(
        title="Klinotaxis Data Visualization",
        template="plotly_white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 6]),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, range=[-0.7, 0.7]
        ),
        plot_bgcolor="white",
        width=800,
        height=500,
        xaxis_scaleanchor="y",
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
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "yanchor": "top",
                "xanchor": "left",
                "x": 0.1,
                "y": -0.1,
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
                    if idx % 10 == 0
                ],
            }
        ],
    )

    return fig
