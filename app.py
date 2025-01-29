import streamlit as st
import numpy as np
import wormsim.worm as worm
import toml
import time
from PIL import Image

# デフォルトの変数の読み込み
config = toml.load("./config.toml")
gene = config["gene"][0]
const = config["const"]
c_mode = config["c_mode"]
concentration_map = config["concentration_map"]

c_elegans = worm.Worm(gene, const, c_mode, concentration_map)

# StreamlitのUI設定
st.header("*C. elegans* Simulator")
st.write(
    """
この線虫シミュレーターは、*Caenorhabditis elegans*（*C. elegans*）が示す塩濃度記憶に依存した塩走性を再現するために作成されました。モデルは以下の論文に基づき構築されています。

Hironaka, M., & Sumi, T. (2024). *A neural network model that generates salt concentration memory-dependent chemotaxis in Caenorhabditis elegans*. [DOI: 10.1101/2024.11.04.621960](https://doi.org/10.1101/2024.11.04.621960)
"""
)

st.info(
    """
使用方法
1. 線虫の個体や塩濃度関数を選択します。
2. 必要に応じて塩濃度などのパラメータを調整します。  
3. [シミュレーションを実行]を押すことで、画面下部にアニメーションが表示されます。
""",
    icon="📖",
)

plot = st.empty()

with st.container(height=400):
    col1, col2 = st.columns([3, 5])
    with col1:
        select_gene = st.radio(
            "線虫の個体を選択してください。",
            ["高塩濃度育成", "低塩濃度育成"],
            help="線虫は培養中に記憶した塩濃度に基づき、現在の環境における嗜好行動を示します。",
        )
    with col2:
        st.write("神経回路：")
        if select_gene == "高塩濃度育成":
            c_elegans.gene = config["gene"][0]
            image = Image.open("./image/connectome_high.png")
        elif select_gene == "低塩濃度育成":
            c_elegans.gene = config["gene"][1]
            image = Image.open("./image/connectome_low.png")

        with st.expander("画像を表示"):
            st.image(
                image,
                caption="白い円は化学感覚ニューロン、灰色の円は介在ニューロン、黒い円は運動ニューロンを表しています。青い矢印と赤い平らな矢印は、それぞれ興奮性と抑制性のシナプス接続を示しています。緑の線は電気的ギャップ結合を示しています。また、接続の太さはそれぞれの結合の強度を示しています。",
                use_container_width=True,
            )

    col1, col2 = st.columns([3, 5])
    with col1:
        select_c_mode = st.radio(
            "塩濃度関数を選択してください。",
            ["ガウス分布１", "ガウス分布２"],
            help="論文中では、ガウス分布１の関数のみ使用されています。",
        )
    with col2:
        st.write("塩濃度関数：")
        if select_c_mode == "ガウス分布１":
            c_elegans.c_mode = 1
            c_elegans.color_scheme = config["concentration_map"]["color_scheme_blue"]
            st.write(
                "$C(x,y)=C_0e^{-\\frac{(x-x_{peak})^2+(y-y_{peak})^2}{2\\lambda^2}}$"
            )
        elif select_c_mode == "ガウス分布２":
            c_elegans.c_mode = 2
            c_elegans.color_scheme = config["concentration_map"][
                "color_scheme_red_blue"
            ]
            st.write(
                "$C(x,y)=C_0[e^{-\\frac{(x-x_{peak})^2+(y-y_{peak})^2}{2\\lambda^2}}-e^{-\\frac{(x+x_{peak})^2+(y+y_{peak})^2}{2\\lambda^2}}]$"
            )

    with st.expander("塩濃度の設定", icon="⚙️"):
        col1, col2 = st.columns(2)
        with col1:
            c_elegans.x_peak = st.slider(
                "$x_{peak}$ /cm",
                min_value=0.0,
                max_value=10.0,
                value=const["x_peak"],
                step=0.1,
                help="Gradient Peakのx座標",
            )
            c_elegans.y_peak = st.slider(
                "$y_{peak}$ /cm",
                min_value=-5.0,
                max_value=5.0,
                value=const["y_peak"],
                step=0.1,
                help="Gradient Peakのy座標",
            )
        with col2:
            c_elegans.c_0 = st.slider(
                "$C_0$ /mM",
                min_value=0.0,
                max_value=5.0,
                value=const["c_0"],
                step=0.1,
                help="塩濃度の最大値を決めるパラメータ",
            )
            c_elegans.lambda_ = st.slider(
                "$\\lambda$ /cm",
                min_value=0.0,
                max_value=5.0,
                value=const["lambda"],
                step=0.1,
                help="塩濃度の広がり方を決めるパラメータ",
            )

    with st.expander("その他の設定", icon="⚙️"):
        col1, col2 = st.columns(2)
        with col1:
            c_elegans.mu_0 = st.slider(
                "進行方向 /rad",
                min_value=0.0,
                max_value=2 * np.pi,
                value=const["mu_0"],
                step=0.1,
                help="線虫の初期の進行方向",
            )
        with col2:
            c_elegans.time = st.slider(
                "シミュレーション時間 /s",
                min_value=0.0,
                max_value=500.0,
                value=const["simulation_time"],
                step=1.0,
                help="シミュレーションの時間（実行時間ではない）",
            )
        select_animation = st.radio(
            "アニメーションの描画精度",
            ["低レベル", "中レベル", "高レベル"],
            index=1,
            help="レベルが低いほどフレーム数が減少する（使用しているシミュレーション結果は同じ）",
        )

if select_animation == "低レベル":
    downsampling_factor: int = 300
    animation_duration: int = 80
elif select_animation == "中レベル":
    downsampling_factor: int = 200
    animation_duration: int = 50
elif select_animation == "高レベル":
    downsampling_factor: int = 100
    animation_duration: int = 10

fig = c_elegans.create_concentration_map()
plot.plotly_chart(fig)


# ボタンを配置
if st.button(
    "シミュレーションを実行", type="primary", use_container_width=True, icon="💻"
):
    start_time = time.time()
    with st.spinner("実行中..."):
        trajectory = c_elegans.klinotaxis_rs()
        fig = c_elegans.create_klintaxis_animation(
            trajectory=trajectory,
            downsampling_factor=downsampling_factor,
            animation_duration=animation_duration,
        )
        st.success("画面左下にある&#9654;からアニメーションを再生できます。", icon="🔽")
        st.plotly_chart(fig)
    end_time = time.time()

    # 経過時間を表示
    total_time = end_time - start_time
    st.toast(
        f"シミュレーションが終わりました（{total_time:.2f} s）。画面を下にスクロールしてください。"
    )
