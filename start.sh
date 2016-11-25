#!/bin/bash

# Function to append the name of this script to output messages.
# $1: Message (in quotes)
# Returns nothing, but prints to stdout.
function echo_start {
    echo "[start.sh] $1"
}

# Function to try to exit elegantly. Returns nothing, but exits the bash
# process.
function abort {
    echo_start "Aborting 'start' operation by running 'stop'..."
    ./stop.sh
    exit 1
}

# Function to create a BuildBot worker (slave).
# $1: Relative path (directory) to create the worker in, which does not exist.
# $2: Worker name
# $3: Worker access password
# Returns nothing.
function create_buildbot_worker {
    echo_start "Creating worker $2..."
    if [ -d "$1" ]; then
        echo_start "Error in creating worker: directory $1 already exists."
        abort
    fi

    local CREATE_OUTERR="$($BUILDBOT_EXEC_PREFIX/buildbot-worker create-worker "$1" localhost:9990 "$2" "$3")"
    if [ $? -ne 0 ]; then
        echo_start "Error in creating worker:"
        echo_start "$CREATE_OUTERR"
        abort
    else
        echo_start "Worker $2 created."
    fi
}

# Find buildbot path. Optionally, user can define environment variable
# $BUILDBOT_EXEC_PREFIX themselves.
if [ -z "$BUILDBOT_EXEC_PREFIX" ]; then
    if [ -z "$(which buildbot)" ]; then
        echo_start "Error: command 'buildbot' not found."
    else
        BUILDBOT_EXEC_PREFIX="$(dirname "$(which buildbot)")"
    fi
fi

# Create transient files for master (database, etc.)
echo_start "Preparing master for start using upgrade-master..."
UPGRADE_OUTERR="$($BUILDBOT_EXEC_PREFIX/buildbot upgrade-master master/ 2>&1)"
if [ $? -ne 0 ]; then
    echo_start "Error in preparing master:"
    echo_start "$UPGRADE_OUTERR"
    abort
else
    echo_start "Master prepared."
fi

# Create workers.
for zI in $(seq 1 2); do
    create_buildbot_worker worker_$zI worker_$zI pass_$zI
done

# Start the master, but give up if there is a problem.
echo_start "Starting the BuildBot master..."
START_OUTERR="$($BUILDBOT_EXEC_PREFIX/buildbot start master/ 2>&1)"
if [ $? -ne 0 ]; then
    echo_start "Error in starting master:"
    echo_start "$START_OUTERR"
    abort
else
    echo_start "Master started."
fi

# Start the workers, but give up if there is a problem.
for zI in $(seq 1 2); do
    echo_start "Starting worker worker_$zI..."
    WORKER_START_OUTERR="$($BUILDBOT_EXEC_PREFIX/buildbot-worker start worker_$zI 2>&1)"
    if [ $? -ne 0 ]; then
        echo_start "Error in starting worker worker_$zI:"
        echo_start "$WORKER_START_OUTERR"
        abort
    else
        echo_start "Worker worker_$zI started."
    fi
done
