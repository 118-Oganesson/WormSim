import streamlit as st
import numpy as np
import wormsim.worm as worm
import toml
import time

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
    "この線虫シミュレーターは、*Caenorhabditis elegans*（*C. elegans*）が示す塩濃度記憶に依存した塩走性を再現するために作成されました。モデルは以下の論文を基に構築されています。"
)
st.write(
    "Hironaka, M., & Sumi, T. (2024). A neural network model that generates salt concentration memory-dependent chemotaxis in Caenorhabditis elegans. [DOI: 10.1101/2024.11.04.621960](https://doi.org/10.1101/2024.11.04.621960)"
)

select_gene = st.selectbox(
    "使用する線虫の個体を選択してください。", ["高塩濃度育成", "低塩濃度育成"]
)
if select_gene == "高塩濃度育成":
    c_elegans.gene = config["gene"][0]
elif select_gene == "低塩濃度育成":
    c_elegans.gene = config["gene"][1]

select_c_mode = st.selectbox(
    "使用する塩濃度関数を選択してください。", ["関数１", "関数２"]
)
if select_c_mode == "関数１":
    c_elegans.c_mode = 1
    c_elegans.color_scheme = config["concentration_map"]["color_scheme_blue"]
    st.write("$C(x,y)=C_0e^{-\\frac{(x-x_{peak})^2+(y-y_{peak})^2}{2\\lambda^2}}$")
elif select_c_mode == "関数２":
    c_elegans.c_mode = 2
    c_elegans.color_scheme = config["concentration_map"]["color_scheme_red_blue"]
    st.write(
        "$C(x,y)=C_0[e^{-\\frac{(x-x_{peak})^2+(y-y_{peak})^2}{2\\lambda^2}}-e^{-\\frac{(x+x_{peak})^2+(y+y_{peak})^2}{2\\lambda^2}}]$"
    )

with st.expander("塩濃度の設定"):
    col1, col2 = st.columns([1, 1])
    with col1:
        c_elegans.x_peak = st.slider(
            "$x_{peak}$ /cm",
            min_value=0.0,
            max_value=10.0,
            value=const["x_peak"],
            step=0.1,
        )
        c_elegans.y_peak = st.slider(
            "$y_{peak}$ /cm",
            min_value=-5.0,
            max_value=5.0,
            value=const["y_peak"],
            step=0.1,
        )
    with col2:
        c_elegans.c_0 = st.slider(
            "$C_0$ /mM",
            min_value=0.0,
            max_value=5.0,
            value=const["c_0"],
            step=0.1,
        )
        c_elegans.lambda_ = st.slider(
            "$\\lambda$ /cm",
            min_value=0.0,
            max_value=5.0,
            value=const["lambda"],
            step=0.1,
        )

with st.expander("その他の設定"):
    col1, col2 = st.columns([1, 1])
    with col1:
        c_elegans.mu_0 = st.slider(
            "direction of movement /rad",
            min_value=0.0,
            max_value=2 * np.pi,
            value=const["mu_0"],
            step=0.1,
        )
    with col2:
        c_elegans.time = st.slider(
            "simulation time /s",
            min_value=0.0,
            max_value=500.0,
            value=const["simulation_time"],
            step=1.0,
        )

concentration_plot = st.empty()
fig = c_elegans.create_concentration_map()
st.plotly_chart(fig)


# ボタンを配置
if st.button("Run Simulation"):
    start_time = time.time()
    with st.spinner("Calculating trajectory..."):
        trajectory = c_elegans.klinotaxis_rs()
    mid_time = time.time()
    with st.spinner("Creating animation..."):
        fig = c_elegans.create_klintaxis_animation(trajectory)
        st.plotly_chart(fig)
    end_time = time.time()

    # 経過時間を表示
    calculation_time = mid_time - start_time
    animation_time = end_time - mid_time
    total_time = end_time - start_time

    st.write(f"Calculation time: {calculation_time:.2f} seconds")
    st.write(f"Animation creation time: {animation_time:.2f} seconds")
    st.write(f"Total time: {total_time:.2f} seconds")
