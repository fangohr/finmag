#!/bin/bash

# Script that stops the master, and attempts to tear down the workers.

# Function to append the name of this script to output messages.
# $1: Message (in quotes)
# Returns nothing, but prints to stdout.
function echo_stop {
    echo "[stop.sh] $1"
}

# Find buildbot path. Optionally, user can define environment variable
# $BUILDBOT_EXEC_PREFIX themselves.
if [ -z "$BUILDBOT_EXEC_PREFIX" ]; then
    if [ -z "$(which buildbot)" ]; then
        echo_stop "Error: command 'buildbot' not found."
    else
        BUILDBOT_EXEC_PREFIX="$(dirname "$(which buildbot)")"
    fi
fi

echo_stop "Stopping master..."
$BUILDBOT_EXEC_PREFIX/buildbot stop master/
echo_stop "Master stopped"

# Stopping and destroying workers.
for zI in $(seq 1 2); do
    if [ ! -d worker_$zI ]; then
        echo_stop "Worker worker_$zI not found. Not destroying."
    fi
    echo_stop "Stopping worker_$zI..."
    $BUILDBOT_EXEC_PREFIX/buildbot-worker stop worker_$zI
    echo_stop "Destroying worker_$zI..."
    rm -rf worker_$zI
done
