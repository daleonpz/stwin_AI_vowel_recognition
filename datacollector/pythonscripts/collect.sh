#!/bin/bash


# read number of samples from input argument 
# (if not provided, default to 100)
# (if provided, check if it is a number)
# (if not a number, default to 100)
# (if a number, check if it is positive)
# (if not positive, default to 100)
# (if positive, use it)
if [ -z "$1" ]; then
    number_of_samples=20
else
    if ! [[ "$1" =~ ^[0-9]+$ ]]; then
        number_of_samples=20
    else
        if [ "$1" -le 0 ]; then
            number_of_samples=20
        else
            number_of_samples="$1"
        fi
    fi
fi

cd data/ || return
# create a array of labels
labels=( "A" "E" "I" "O" "U" )

for label in "${labels[@]}"; do
    echo "####################################################"
    echo "Alert!!! Data collection for label: $label"
    echo "Press enter to start recording"
    read -r

    # create a directory for each label
    folder_name="vowel_${label}"
    mkdir -p "${folder_name}"

    python3 ../collect_data_new.py  --label "${label}" --samples "${number_of_samples}"
    # move all files with the label to the directory
    mv "$label"* "${folder_name}"

    ## two blank lines between each label
    echo ""
    echo ""
    echo "Collecting data for label: $label done"
done

echo "All done"

cd .. || exit # From data
