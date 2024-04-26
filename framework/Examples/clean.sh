echo =====================================
echo ====== Deleting old data files ======
echo
rm CMakeCache.txt

rm -f Model*
rm -f slurm*
rm -f summary*

rm -f Case00/proposal-*
rm -f Case00/map-*
rm -f Case00/burnin-*
rm -f Case00/Model*
rm -f Case00/Info-*

rm -f Case00/burnin/*.dat
rm -f Case00/chains/*.dat
rm -f Case00/evidence/*.dat

