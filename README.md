# atmacup10_colum2131_tubo
atmaCup#10 提出用リポジトリ

## requirements
* Docker 20.10.5
* docker-compose 1.25.0
* NVIDIA Docker 2.5.0

### build enviroment 
<pre>
docker-compose up -d
</pre>


## directory tree
<pre>
.
├── data                     <---- 生データ
├── ensemble                 <---- ensemble用のoof_pred,test_pred
├── features                 <---- 作った特徴量
└── src                      <---- モデル実行,特徴量作成用の.ipynb,.py
    ├── config               <---- modelのparm
    ├── create_dataset       <---- 特徴量作成用の.py
    ├── module               <---- model定義,BaseBlockで作った特徴量作成module
    ├── tubo_model1.ipynb    <---- stacking１段目のmodel作成(特徴量作成もこのファイルでできる）
    └── tubo_stacking.ipynb  <---- stacking → submission.csv作成
</pre>

## Usage
### create_dataset
すでに作ってあるので飛ばしても大丈夫。
- tubo
<pre>
cd src/tubo/create_dataset
python run.py
</pre>
- colum2131
<pre>
cd src/colum2131/create_dataset
hogehoge
</pre>
### stage1_model train
ここも学習結果はすでに保存してある。
- tubo
    - src/tubo/tubo_model1.ipynbを上から実行
- colum2131
    - src/colum2131/model1.ipynbを上から実行
    - src/colum2131/model2.ipynbを上から実行
### ensemble
stackingして予測結果の保存
- src/tubo/tubo_stacking.ipynbを上から実行

## Author
* tubo
* colum2131
