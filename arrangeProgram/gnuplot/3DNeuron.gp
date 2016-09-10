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
set yl "neuron2" rotate by 90
set yr [0:1]
set zl "neuron3"
set zr [0:1]
set key outside
set format "%5.1f"

set tics offset 0, 0, 0

# view 0, 0
x = 0; y = 0
set view x, y
set ytics offset -26.5, 0, 0; set yl ""; set label 1 "neuron2" at -0.2, 0.5, 0 center rotate by 90
set ytics rotate by 90
unset ztics; unset zl
cd sprintf("%s/outNeuron/%s/view%d,%d", path, subDirName, x, y)
set terminal png font "Arial,15"
set out sprintf("out_neuron%d_view%d,%d.png", n, x, y)
set title sprintf("out_neuron%d_%s_view%d,%d", n, subDirName, x, y)
splot sprintf("%s/middle/%s/middle%d.dat", path, subDirName, n) w lp lw 2 title "locus"

# view 90, 0
x = 90; y = 0
set view x, y
set xtics offset 0, -1, 0
set xl "neuron1" 0, -2, 0
  set ztics;   set zl "neuron3" 1.0, 0, 0 rotate by 90
unset ytics; unset yl; unset label 1
cd sprintf("%s/outNeuron/%s/view%d,%d", path, subDirName, x, y)
set terminal png font "Arial,15"
set out sprintf("out_neuron%d_view%d,%d.png", n, x, y)
set title sprintf("out_neuron%d_%s_view%d,%d", n, subDirName, x, y)
splot sprintf("%s/middle/%s/middle%d.dat", path, subDirName, n) w lp lw 2 title "locus"

# view 90, 90
x = 90; y = 90
set view x, y
set ytics offset 0, -1, 0; set yl "neuron2" 0, -2, 0
set zl "neuron3" -5, 0, 0
unset xtics; unset xl
cd sprintf("%s/outNeuron/%s/view%d,%d", path, subDirName, x, y)
set terminal png font "Arial,15"
set out sprintf("out_neuron%d_view%d,%d.png", n, x, y)
set title sprintf("out_neuron%d_%s_view%d,%d", n, subDirName, x, y)
splot sprintf("%s/middle/%s/middle%d.dat", path, subDirName, n) w lp lw 2 title "locus"

# view 45, 45
x = 45; y = 45
set view x, y
set xtics offset -0.5, -0.1, 0; set label 1 "neuron1" at 0.3, 0.0, -1.0 center rotate by -33
set ytics offset 1.0, 0, 0;  set yl "";  set label 2 "neuron2" at 0.5, 1.2, -2.1 center rotate by 32
set zl "neuron3" 2, 2.5, 0 rotate by 0
cd sprintf("%s/outNeuron/%s/view%d,%d", path, subDirName, x, y)
set terminal png font "Arial,15"
set out sprintf("out_neuron%d_view%d,%d.png", n, x, y)
set title sprintf("out_neuron%d_%s_view%d,%d", n, subDirName, x, y)
splot sprintf("%s/middle/%s/middle%d.dat", path, subDirName, n) w lp lw 2 title "locus"

unset label 1
unset label 2

# 最初だけの処理
if (n == 0) \
    n = 1; step = 1; cnt = 1; reread   # n = 1にして再スタート

n = n + step
cnt = cnt + 1
if (cnt >= 10) \
    cnt = 1; step = 10 * step

if (n <= limit) \
    reread

