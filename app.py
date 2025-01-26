import streamlit as st
import numpy as np
import wormsim.worm as worm
import toml
import time

# 変数の定義
config = toml.load("./config.toml")
gene = config["gene"]
const = config["const"]
c_mode = config["c_mode"]
concentration_map = config["concentration_map"]

# 線虫
c_elegans = worm.Worm(gene, const, c_mode, concentration_map)
c_elegans.concentration_num = 100

# StreamlitのUI設定
st.title("C. elegans Simulator")
st.header("Concentration Setting")
col1, col2 = st.columns([3, 1])
with col2:
    c_elegans.x_peak = st.slider(
        "X",
        min_value=0.0,
        max_value=10.0,
        value=const["x_peak"],
        step=0.1,
    )
    c_elegans.y_peak = st.slider(
        "Y",
        min_value=-5.0,
        max_value=5.0,
        value=const["y_peak"],
        step=0.1,
    )
    c_elegans.c_0 = st.slider(
        "C_0",
        min_value=0.0,
        max_value=5.0,
        value=const["c_0"],
        step=0.1,
    )
    c_elegans.lambda_ = st.slider(
        "lambda",
        min_value=0.0,
        max_value=5.0,
        value=const["lambda"],
        step=0.1,
    )
    c_elegans.mu_0 = st.slider(
        "mu_0",
        min_value=0.0,
        max_value=2 * np.pi,
        value=const["mu_0"],
        step=0.1,
    )

with col1:
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
