#!/bin/bash

device_uuid="$1"
property="$2"
value="$3"
count="$4"
wait="$5"

base_url="https://api.moehlenhoff.q-loud.de"
user='moe1:password'
# base_url="https://api.opencloudservice.com"
# user='qbeyond_user:G8QWRgvP'

url="${base_url}/devices/${device_uuid}/actions"
payload="{ \
    \"${property}\":${value}
}"

rm -f ./datendatei

echo "Sending $count packages"
for i in $(seq 1 $count);
do
    curl -k -s -u ${user} -H 'Content-Type:application/json' --data-raw "${payload}" ${url} >> ./datendatei
    echo >> ./datendatei
done

echo "Sleeping for ${wait}s"
sleep $wait

array=($(cat "./datendatei"))
for i in ${!array[@]}
do
    action_key=$(echo "${array[$i]}" | jq '."action-key"' -r)
    status=$(curl -s -k -u ${user} -H 'Content-Type:application/json' "${url}/${action_key}" | jq '."status"' -r)
    echo "$action_key: $status"
done

#rm -f ./datendatei


