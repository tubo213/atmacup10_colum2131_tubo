# atmacup10_colum2131_tubo
atmaCup#10 提出用リポジトリ

- directory tree
<pre>
.
├── data                         <---- 生データ
├── ensemble                     <---- ensemble用のoof_pred,test_pred
├── features                     <---- 作った特徴量
└── src                          <---- モデル実行,特徴量作成用の.ipynb,.py
    └── tubo
        ├── PaletteEmbedding     <---- Arai_feature作るのときのmodelの履歴
        ├── config               <---- modelのparm
        ├── create_dataset       <---- 特徴量作成用の.py
        ├── module               <---- model定義,BaseBlockで作った特徴量作成module
        ├── tubo_model1.ipynb    <---- stacking１段目のmodel作成(特徴量作成もこのファイルでできる）
        └── tubo_stacking.ipynb  <---- stacking → submission.csv作成
</pre>

