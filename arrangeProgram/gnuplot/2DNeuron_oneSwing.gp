
set xl "neuron1"
set xr [0:1]
set yl "neuron2"
set yr [0:1]
set key outside
set format "%5.1f"

set tics offset 0, 0, 0

cd path
set terminal png font "Arial,15"
set out sprintf("%s.png", fileName)
set title titleName;
plot sprintf("%s/%s.dat", path, fileName) w lp lw 2 title "locus"


