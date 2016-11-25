#!/bin/bash
BUILDBOT_EXEC_PREFIX="$HOME/miniconda2/bin"
$BUILDBOT_EXEC_PREFIX/buildbot stop master/
$BUILDBOT_EXEC_PREFIX/buildbot-worker stop worker_1/
$BUILDBOT_EXEC_PREFIX/buildbot-worker stop worker_2/
rm -rf worker_1/ worker_2/
