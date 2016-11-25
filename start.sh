#!/bin/bash
BUILDBOT_EXEC_PREFIX="$HOME/miniconda2/bin"

# Create transient files for master (database, etc.)
$BUILDBOT_EXEC_PREFIX/buildbot upgrade-master master/

# Create workers.
$BUILDBOT_EXEC_PREFIX/buildbot-worker create-worker worker_1 localhost:9990 worker_1 pass_1
$BUILDBOT_EXEC_PREFIX/buildbot-worker create-worker worker_2 localhost:9990 worker_2 pass_2

# Start the master and the workers.
$BUILDBOT_EXEC_PREFIX/buildbot start master/
$BUILDBOT_EXEC_PREFIX/buildbot-worker start worker_1/
$BUILDBOT_EXEC_PREFIX/buildbot-worker start worker_2/
