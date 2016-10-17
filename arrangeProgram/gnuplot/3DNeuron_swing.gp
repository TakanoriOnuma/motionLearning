# ���[�v�̐錾
if (exist("n") == 0 || n < 0) \
    n = 0;   # �ϐ��̏�����
if (exist("swingNum") == 0 || swingNum < 0) \
    swingNum = 0;

if (exist("limit") == 0) \
    limit = 1000    # �ϐ��̏�����(�w�K�̉�)
if (exist("path") == 0) \
    path = "/Users/Takanori/Desktop/gp442win32/gnuplot/binary/dats"
if (exist("subDirName") == 0) \
    subDirName = "train"
if (exist("swingLimit") == 0) \
    swingLimit = 100

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
cd sprintf("%s/outNeuron/%s/swing%d/view%d,%d", path, subDirName, swingNum, x, y)
set terminal png font "Arial,15"
set out sprintf("out_neuron%d_view%d,%d.png", n, x, y)
set title sprintf("out_neuron%d_%s/%d_view%d,%d", n, subDirName, swingNum, x, y)
splot sprintf("%s/middle/%s/swing%d/middle%d.dat", path, subDirName, swingNum, n) w lp lw 2 title "locus"

# view 90, 0
x = 90; y = 0
set view x, y
set xtics offset 0, -1, 0
set xl "neuron1" 0, -2, 0
  set ztics;   set zl "neuron3" 1.0, 0, 0 rotate by 90
unset ytics; unset yl; unset label 1
cd sprintf("%s/outNeuron/%s/swing%d/view%d,%d", path, subDirName, swingNum, x, y)
set terminal png font "Arial,15"
set out sprintf("out_neuron%d_view%d,%d.png", n, x, y)
set title sprintf("out_neuron%d_%s/%d_view%d,%d", n, subDirName, swingNum, x, y)
splot sprintf("%s/middle/%s/swing%d/middle%d.dat", path, subDirName, swingNum, n) w lp lw 2 title "locus"

# view 90, 90
x = 90; y = 90
set view x, y
set ytics offset 0, -1, 0; set yl "neuron2" 0, -2, 0
set zl "neuron3" -5, 0, 0
unset xtics; unset xl
cd sprintf("%s/outNeuron/%s/swing%d/view%d,%d", path, subDirName, swingNum, x, y)
set terminal png font "Arial,15"
set out sprintf("out_neuron%d_view%d,%d.png", n, x, y)
set title sprintf("out_neuron%d_%s/%d_view%d,%d", n, subDirName, swingNum, x, y)
splot sprintf("%s/middle/%s/swing%d/middle%d.dat", path, subDirName, swingNum, n) w lp lw 2 title "locus"

# view 45, 45
x = 45; y = 45
set view x, y
set xtics offset -0.5, -0.1, 0; set label 1 "neuron1" at 0.3, 0.0, -1.0 center rotate by -33
set ytics offset 1.0, 0, 0;  set yl "";  set label 2 "neuron2" at 0.5, 1.2, -2.1 center rotate by 32
set zl "neuron3" 2, 2.5, 0 rotate by 0
cd sprintf("%s/outNeuron/%s/swing%d/view%d,%d", path, subDirName, swingNum, x, y)
set terminal png font "Arial,15"
set out sprintf("out_neuron%d_view%d,%d.png", n, x, y)
set title sprintf("out_neuron%d_%s/%d_view%d,%d", n, subDirName, swingNum, x, y)
splot sprintf("%s/middle/%s/swing%d/middle%d.dat", path, subDirName, swingNum, n) w lp lw 2 title "locus"

unset label 1
unset label 2

# �ŏ������̏���
if (n == 0) \
    n = 1; step = 1; cnt = 1; reread   # n = 1�ɂ��čăX�^�[�g

n = n + step
cnt = cnt + 1
if (cnt >= 10) \
    cnt = 1; step = 10 * step

# n��limit�ɒB������swingNum��+1�ɂ���n�����ɖ߂�
if (n > limit) \
    n = 0; swingNum = swingNum + 1
if (swingNum < swingLimit) \
    reread

