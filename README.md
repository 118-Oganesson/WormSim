# 線虫シミュレーター

この[線虫シミュレーター](https://wormsim.streamlit.app/)は、*Caenorhabditis elegans*（*C. elegans*）が示す塩濃度記憶に依存した塩走性を再現するために作成されました。モデルは以下の論文に基づき構築されています。

**参考論文**:  
Hironaka Masakatsu, Sumi Tomonari “A neural network model that generates salt concentration memory-dependent chemotaxis in *Caenorhabditis elegans*” *eLife* **14**:RP104456 (2025).
[https://elifesciences.org/articles/104456](https://elifesciences.org/articles/104456)

このシミュレーターは、Webアプリケーションとして提供されており、視覚的にわかりやすい形で入力パラメータを設定し、線虫の動きをアニメーションとして観察することができます。

## 使用方法

1. [シミュレーター](https://wormsim.streamlit.app/)を開き、下のタブから、線虫の個体や塩濃度関数を選択します。
2. 必要に応じてパラメータを調整します。  
3. **シミュレーションを実行**を押すことで、画面上にアニメーションが表示されます。

## 技術的な特徴

- **Rust**で実装された[シミュレーションエンジン](https://github.com/118-Oganesson/wormsim_rs)を使用することで計算時間を大幅に削減しています。  
- **Plotly**によって、インタラクティブなアニメーションを生成しています。
- **Streamlit**によって、入力や結果を簡単に操作できるシンプルなUIを搭載しました。

線虫の行動メカニズムを視覚的に理解し、学術研究や教育、さらには遊び心のある実験に役立てることができます。ぜひお試しください！

## English

For English, please see [README_en.md](./README_en.md)
