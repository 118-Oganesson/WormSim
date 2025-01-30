# 線虫シミュレーター

この線虫シミュレーターは、*Caenorhabditis elegans*（*C. elegans*）が示す塩濃度記憶に依存した塩走性を再現するために作成されました。モデルは以下の論文に基づき構築されています。

**参考論文**:  
Hironaka, M., & Sumi, T. (2024). *A neural network model that generates salt concentration memory-dependent chemotaxis in Caenorhabditis elegans*.  
[DOI: 10.1101/2024.11.04.621960](https://doi.org/10.1101/2024.11.04.621960)

このシミュレーターは、Webアプリケーションとして提供されており、視覚的にわかりやすい形で入力パラメータを設定し、線虫の動きをアニメーションとして観察することができます。

## 使用方法
1. [シミュレーター](https://wormsim.streamlit.app/)を開き、下のタブから、線虫の個体や塩濃度関数を選択します。
2. 必要に応じて塩濃度などのパラメータを調整します。  
3. **シミュレーションを実行**を押すことで、画面上にアニメーションが表示されます。

## 技術的な特徴
- **Rust**で実装された[シミュレーションエンジン](https://github.com/118-Oganesson/wormsim_rs)を使用することで計算時間を大幅に削減しています。  
- **Plotly**によって、インタラクティブなアニメーションを生成しています。 
- **Streamlit**によって、入力や結果を簡単に操作できるシンプルなUIを搭載しました。

線虫の行動メカニズムを視覚的に理解し、学術研究や教育、さらには遊び心のある実験に役立てることができます。ぜひお試しください！


# *C. elegans* Simulator  

This *Caenorhabditis elegans* (*C. elegans*) simulator was developed to reproduce salt chemotaxis behavior based on salt concentration memory. The model is built upon the following research paper:  

**Reference Paper:**  
Hironaka, M., & Sumi, T. (2024). *A neural network model that generates salt concentration memory-dependent chemotaxis in Caenorhabditis elegans*.  
[DOI: 10.1101/2024.11.04.621960](https://doi.org/10.1101/2024.11.04.621960)  

This simulator is available as a web application, allowing users to intuitively configure input parameters and observe *C. elegans* movements through animations.  

## **How to Use**  
1. Open the [simulator](https://wormsim.streamlit.app/) and select *C. elegans* individuals and salt concentration functions from the tabs below.  
2. Adjust parameters such as salt concentration as needed.  
3. Click **Run Simulation** to display the animation on the screen.  

## **Technical Features**  
- A **Rust-based** [simulation engine](https://github.com/118-Oganesson/wormsim_rs) significantly reduces computation time.  
- **Plotly** generates interactive animations.  
- **Streamlit** provides a simple and user-friendly interface for input and result manipulation.  

This simulator helps visualize the behavioral mechanisms of *C. elegans*, making it useful for academic research, education, and even playful experiments. Give it a try!  