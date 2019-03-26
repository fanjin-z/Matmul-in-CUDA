#performance with default thread geometry
./mmpy -n 1024 -r 3 -R

#correctness
./mmpy -n 256 -r 3 -R
./mmpy -n 512 -r 3 -R
./mmpy -n 2048 -r 2 -R
#
#groups of three only
./mmpy -n 789 -r 2 -R
