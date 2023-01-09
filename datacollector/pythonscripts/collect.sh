#!/bin/bash

cd data/ || return

# create a array of labels
labels=( "A" "E" "I" "O" "U" )

for label in "${labels[@]}"; do
    echo "####################################################"
    echo "Collecting data for label: $label"
    # create a directory for each label
    folder_name="vowel_${label}"
    mkdir -p "${folder_name}"

    number_of_samples=2
    for ((i=0; i < number_of_samples; i++)); do
        # collect data for each label
        filenumber=$(printf "%04d" "$i")
        python3 ../collect_data.py -F "${label}_${filenumber}.csv" -l "${label}"
    done
    # move all files with the label to the directory
    mv "$label"* "${folder_name}"
done

echo "All done"

cd .. || exit # From data
