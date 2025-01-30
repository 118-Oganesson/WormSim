import streamlit as st
import numpy as np
import wormsim.worm as worm
import wormsim.language as lang
import toml
from PIL import Image


# デフォルトの変数の読み込み
config = toml.load("./config.toml")
gene = config["gene"][0]
const = config["const"]
c_mode = config["c_mode"]
concentration_map = config["concentration_map"]
c_elegans = worm.Worm(gene, const, c_mode, concentration_map)

lang_dict = lang.language

# StreamlitのUI設定
col1, col2 = st.columns([3, 1])
with col2:
    selected_lang = st.selectbox("🌍 言語 / Language", ["日本語 (JA)", "English (EN)"])
    # 言語コードを判定
    lang_code = "ja" if "日本語" in selected_lang else "en"
with col1:
    st.header(lang_dict[lang_code]["title"])

st.write(lang_dict[lang_code]["description"])
st.write(lang_dict[lang_code]["paper_reference"])
st.info(lang_dict[lang_code]["usage_info"], icon="📖")

st.write(lang_dict[lang_code]['simulation_environment'])
plot = st.empty()

tab1, tab2, tab3 = st.tabs(lang_dict[lang_code]["tab_names"])

with tab1:
    col1, col2 = st.columns([3, 5])
    with col1:
        select_gene = st.radio(
            lang_dict[lang_code]["select_worm_label"],
            lang_dict[lang_code]["select_worm_options"],
            help=lang_dict[lang_code]["select_worm_help"],
        )
    with col2:
        st.write(lang_dict[lang_code]["neural_circuit"])
        if select_gene == lang_dict[lang_code]["select_worm_options"][0]:
            c_elegans.gene = config["gene"][0]
            image = Image.open("./image/connectome_high.png")
        elif select_gene == lang_dict[lang_code]["select_worm_options"][1]:
            c_elegans.gene = config["gene"][1]
            image = Image.open("./image/connectome_low.png")

        with st.expander(lang_dict[lang_code]["expander_image_label"]):
            st.image(
                image,
                caption=lang_dict[lang_code]["image_caption"],
                use_container_width=True,
            )

with tab2:
    col1, col2 = st.columns([3, 5])
    with col1:
        select_c_mode = st.radio(
            lang_dict[lang_code]["select_concentration_label"],
            lang_dict[lang_code]["select_concentration_options"],
            help=lang_dict[lang_code]["select_concentration_help"],
        )
    with col2:
        st.write(lang_dict[lang_code]["concentration_function"])
        if select_c_mode == lang_dict[lang_code]["select_concentration_options"][0]:
            c_elegans.c_mode = 1
            c_elegans.color_scheme = config["concentration_map"]["color_scheme_blue"]
            st.write(lang_dict[lang_code]["concentration_function_1"])
        elif select_c_mode == lang_dict[lang_code]["select_concentration_options"][1]:
            c_elegans.c_mode = 2
            c_elegans.color_scheme = config["concentration_map"][
                "color_scheme_red_blue"
            ]
            st.write(lang_dict[lang_code]["concentration_function_2"])

    col1, col2 = st.columns(2)
    with col1:
        c_elegans.x_peak = st.slider(
            lang_dict[lang_code]["slider_x_peak"],
            min_value=0.0,
            max_value=10.0,
            value=const["x_peak"],
            step=0.1,
            help=lang_dict[lang_code]["slider_x_peak_help"],
        )
        c_elegans.y_peak = st.slider(
            lang_dict[lang_code]["slider_y_peak"],
            min_value=-5.0,
            max_value=5.0,
            value=const["y_peak"],
            step=0.1,
            help=lang_dict[lang_code]["slider_y_peak_help"],
        )
    with col2:
        c_elegans.c_0 = st.slider(
            lang_dict[lang_code]["slider_c_0"],
            min_value=0.0,
            max_value=5.0,
            value=const["c_0"],
            step=0.1,
            help=lang_dict[lang_code]["slider_c_0_help"],
        )
        c_elegans.lambda_ = st.slider(
            lang_dict[lang_code]["slider_lambda"],
            min_value=0.0,
            max_value=5.0,
            value=const["lambda"],
            step=0.1,
            help=lang_dict[lang_code]["slider_lambda_help"],
        )

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        c_elegans.mu_0 = st.slider(
            lang_dict[lang_code]["slider_mu_0"],
            min_value=0.0,
            max_value=2 * np.pi,
            value=const["mu_0"],
            step=0.1,
            help=lang_dict[lang_code]["slider_mu_0_help"],
        )
    with col2:
        c_elegans.time = st.slider(
            lang_dict[lang_code]["slider_simulation_time"],
            min_value=0.0,
            max_value=1000.0,
            value=const["simulation_time"],
            step=1.0,
            help=lang_dict[lang_code]["slider_simulation_time_help"],
        )
    select_animation = st.radio(
        lang_dict[lang_code]["animation_quality_label"],
        lang_dict[lang_code]["animation_quality_options"],
        index=1,
        help=lang_dict[lang_code]["animation_quality_help"],
    )

if select_animation == lang_dict[lang_code]["animation_quality_options"][0]:
    downsampling_factor: int = 300
    animation_duration: int = 80
elif select_animation == lang_dict[lang_code]["animation_quality_options"][1]:
    downsampling_factor: int = 200
    animation_duration: int = 50
elif select_animation == lang_dict[lang_code]["animation_quality_options"][2]:
    downsampling_factor: int = 100
    animation_duration: int = 10

fig = c_elegans.create_concentration_map()
plot.plotly_chart(fig, config={"staticPlot": False})


@st.dialog(lang_dict[lang_code]["dialog_title"], width="large")
def simulation():
    with st.spinner(lang_dict[lang_code]["spinner_message"]):
        trajectory = c_elegans.klinotaxis_rs()
        fig = c_elegans.create_klintaxis_animation(
            trajectory=trajectory,
            downsampling_factor=downsampling_factor,
            animation_duration=animation_duration,
        )
        st.success(lang_dict[lang_code]["success_message"], icon="🔽")
        st.plotly_chart(fig)


# ボタンを配置
if st.button(
    lang_dict[lang_code]["simulation_button"],
    type="primary",
    use_container_width=True,
    icon="💻",
):
    simulation()
