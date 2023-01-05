#!/bin/bash

cd data/ || return
foldername="rasiermaschine_50_rpm_anomalities"
mkdir $foldername 
cd $foldername || return
counter=0
while [ $counter -le 10 ]
do
    echo $counter
    counter_str=$(printf '%04i\n' $counter)
    python ../collect_data.py -F "$counter_str".csv
    ((counter++))
done

echo "All done"

cd .. || exit # From ${foldername}
cd .. || exit # From data
