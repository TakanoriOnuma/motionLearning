# ���[�v�̐錾
if (exist("n") == 0 || n < 0) \
    n = 0;   # �ϐ��̏�����

if (exist("limit") == 0) \
    limit = 1000    # �ϐ��̏�����(�w�K�̉�)
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

# �ŏ������̏���
if (n == 0) \
    n = 1; step = 1; cnt = 1; reread   # n = 1�ɂ��čăX�^�[�g

n = n + step
cnt = cnt + 1
if (cnt >= 10) \
    cnt = 1; step = 10 * step

if (n <= limit) \
    reread

