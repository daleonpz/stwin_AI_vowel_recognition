#!/bin/bash

device_uuid="$1"
datei="$2"

base_url="https://api.moehlenhoff.q-loud.de"
user='moe1:password'
# base_url="https://api.opencloudservice.com"
# user='qbeyond_user:G8QWRgvP'

url="${base_url}/devices/${device_uuid}/actions"

array=($(cat "${datei}"))
for i in ${!array[@]}
do
    action_key=$(echo "${array[$i]}" | jq '."action-key"' -r)
    status=$(curl -s -k -u ${user} -H 'Content-Type:application/json' "${url}/${action_key}" | jq '."status"' -r)
    echo "$action_key: $status"
done


