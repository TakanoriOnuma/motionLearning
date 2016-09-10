if (exist("limit") == 0) \
    limit = 1000    # •Ï”‚Ì‰Šú‰»(ŠwK‰ñ”)
if (exist("path") == 0) \
    path = "/Users/Takanori/Desktop/gp442win32/gnuplot/binary/dats"

cd path
width = 2

set title "error"
set xr [1:limit]
set xl "epoch"
set yl "error"
set logscale x
set logscale y
set grid

set terminal png
set out "error.png"
plot "error.dat" using 1:2 w line lw width title "train"
