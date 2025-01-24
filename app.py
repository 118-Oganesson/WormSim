import streamlit as st
import plotly.graph_objects as go
import wormsim.worm as worm
import wormsim_rs as rs
import time

# 変数の定義
gene = [
    -0.8094022576319283,
    -0.6771492613425638,
    0.05892807075993428,
    -0.4894407617977082,
    0.1593721867510597,
    0.3576592038271041,
    -0.5664294232926526,
    -0.7853343958692636,
    0.6552003805912084,
    -0.6492992485125678,
    -0.5482223848375227,
    -0.956542705465967,
    -1.0,
    -0.7386107983898611,
    0.02074396537515929,
    0.7150315462816783,
    -0.9243504880454858,
    0.1353396882729762,
    0.9494528443702027,
    0.7727883271643218,
    -0.6046043758402895,
    0.7969062294208619,
]

const = {
    "alpha": -0.01,
    "c_0": 1,
    "lambda": 1.61,
    "x_peak": 4.5,
    "y_peak": 0.0,
    "dt": 0.01,
    "periodic_time": 4.2,
    "frequency": 0.033,
    "mu_0": 0,
    "velocity": 0.022,
    "simulation_time": 300,
    "time_constant": 0.1,
}

c_mode = 1

# StreamlitのUI設定
st.title("C. elegans Simulation")

# ボタンを配置
if st.button("Run Simulation Python"):
    # ローディングインジケーター表示
    start_time = time.time()  # 処理開始時間記録
    with st.spinner("Calculating trajectory..."):
        c_elegans = worm.Worm(gene, const, c_mode)
        trajectory = c_elegans.klinotaxis()
    mid_time = time.time()  # 計算後の時間記録
    with st.spinner("Creating animation..."):
        fig = c_elegans.create_klintaxis_animation(trajectory)
        st.plotly_chart(fig)
    end_time = time.time()  # アニメーション作成後の時間記録

    # 経過時間を表示
    calculation_time = mid_time - start_time
    animation_time = end_time - mid_time
    total_time = end_time - start_time

    st.write(f"Calculation time: {calculation_time:.2f} seconds")
    st.write(f"Animation creation time: {animation_time:.2f} seconds")
    st.write(f"Total time: {total_time:.2f} seconds")

# ボタンを配置
if st.button("Run Simulation Rust"):
    # ローディングインジケーター表示
    start_time = time.time()  # 処理開始時間記録
    with st.spinner("Calculating trajectory..."):
        c_elegans = worm.Worm(gene, const, c_mode)
        trajectory = rs.klinotaxis({"gene": gene}, const, c_mode)
    mid_time = time.time()  # 計算後の時間記録
    with st.spinner("Creating animation..."):
        fig = c_elegans.create_klintaxis_animation(trajectory)
        st.plotly_chart(fig)
    end_time = time.time()  # アニメーション作成後の時間記録

    # 経過時間を表示
    calculation_time = mid_time - start_time
    animation_time = end_time - mid_time
    total_time = end_time - start_time

    st.write(f"Calculation time: {calculation_time:.2f} seconds")
    st.write(f"Animation creation time: {animation_time:.2f} seconds")
    st.write(f"Total time: {total_time:.2f} seconds")
