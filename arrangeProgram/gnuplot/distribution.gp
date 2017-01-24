
cd path

set logscale x
set ytics nomirror
set y2tics
set grid

set xl "epoch"
set yl "std"
set y2l "free space"

set title sprintf("%s", titleName)

cd path
set terminal png font "Arial,15"
set out sprintf("%s.png", fileName)
plot sprintf("%s/%s.dat", path, fileName) using 1:2 w lp lw 2 title "std", \
     sprintf("%s/%s.dat", path, fileName) using 1:3 w lp lw 2 title "free space" axes x1y2


