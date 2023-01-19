#!/bin/bash

TEST="all"

echo "###############################"
if [ "$1" != "" ]; then
    echo "Running test: " "$1"
    TEST="$1"
else
    echo "Running all tests"
fi
echo "###############################"

IMAGE_NAME="ruby:2.5"

LOCAL_WORKING_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
DOCKER_WORKING_DIR="/home/myapp"

COMMAND_TO_RUN_ON_DOCKER=(sh -c "gem install ceedling && cd unittest/ && ceedling clobber test:${TEST}")


echo "###############################"
echo -e "Working Directory: 	 ${LOCAL_WORKING_DIR}"
echo "###############################"


docker run \
            --rm \
            -v "${LOCAL_WORKING_DIR}":"${DOCKER_WORKING_DIR}"\
            -w "${DOCKER_WORKING_DIR}"\
            --name my_container_test "${IMAGE_NAME}"  \
            "${COMMAND_TO_RUN_ON_DOCKER[@]}"


