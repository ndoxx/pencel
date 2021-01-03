set title "Loss manifold"
set palette color positive
set samples 50
set isosamples 50
set view map
set xlabel "saturation"
set ylabel "lightness"
splot "manifold.dat" using 1:2:3 with pm3d
unset pm3d
unset view