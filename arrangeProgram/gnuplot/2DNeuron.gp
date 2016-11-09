# ループの宣言
if (exist("n") == 0 || n < 0) \
    n = 0;   # 変数の初期化

if (exist("limit") == 0) \
    limit = 1000    # 変数の初期化(学習の回数)
if (exist("path") == 0) \
    path = "/Users/Takanori/Desktop/gp442win32/gnuplot/binary/dats"
if (exist("subDirName") == 0) \
    subDirName = "train"

set xl "neuron1"
set xr [0:1]
set yl "neuron2"
set yr [0:1]
set key outside
set format "%5.1f"

set tics offset 0, 0, 0

cd sprintf("%s/outNeuron/%s", path, subDirName)
set terminal png font "Arial,15"
set out sprintf("out_neuron%d.png", n)
set title sprintf("out_neuron%d_%s", n, subDirName);
plot sprintf("%s/middle/%s/middle%d.dat", path, subDirName, n) w lp lw 2 title "locus"

# 最初だけの処理
if (n == 0) \
    n = 1; step = 1; cnt = 1; reread   # n = 1にして再スタート

n = n + step
cnt = cnt + 1
if (cnt >= 10) \
    cnt = 1; step = 10 * step

if (n <= limit) \
    reread

