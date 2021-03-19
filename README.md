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
