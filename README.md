# ACRNN_EEG
## Paper:
"EEG based Emotion Recognition via Channel-wise Attention and Self Attention"

## environment
docker + anaconda + pytorch + GPU
1. GPU setting

2. docker <br>
docker imageの取得
```
docker pull pytorch/pytorch
docker run -it --name 任意のコンテナ名 -v マウントしたいディレクトリ：マウント先のパス イメージの名前：イメージのタグ /bin/bash
```
3. library install
```
pip install scipy # .mat fileを読み込むため
```
4. git install
```
apt-get update && apt-get install git
```
5. tmux
 作成途中

## Dataset
- DEAP <br>
http://www.eecs.qmul.ac.uk/mmv/datasets/deap/
- DREAMER <br>
https://zenodo.org/record/546113#.YIb00n2mMTU

## Training
```
pyhon main.py
```


## Reference
- docker + gpu + pytorch <br>
https://qiita.com/conankonnako/items/787b69cd8cbfe7d7cb88
- similarity code (tensorflow)
https://github.com/Chang-Li-HFUT/ACRNN
