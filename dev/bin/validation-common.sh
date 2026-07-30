#!/usr/bin/env bash

# Shared guard for read-only validation commands. It compares full tracked
# diffs, so intentional edits that predate the command are preserved. [Codex GPT-5]
install_tracked_files_guard() {
    validation_repo_root="$1"
    validation_label="$2"
    validation_tracked_before="$(mktemp)"
    validation_tracked_after="$(mktemp)"
    git -C "${validation_repo_root}" diff --binary --no-ext-diff HEAD -- \
        > "${validation_tracked_before}"

    trap check_tracked_files_unchanged EXIT
}

check_tracked_files_unchanged() {
    command_status=$?
    trap - EXIT
    git -C "${validation_repo_root}" diff --binary --no-ext-diff HEAD -- \
        > "${validation_tracked_after}"
    if ! cmp -s "${validation_tracked_before}" "${validation_tracked_after}"; then
        echo "ERROR: ${validation_label} changed tracked files." >&2
        diff -u --label tracked-before --label tracked-after \
            "${validation_tracked_before}" "${validation_tracked_after}" >&2 || true
        rm -f "${validation_tracked_before}" "${validation_tracked_after}"
        exit 1
    fi
    rm -f "${validation_tracked_before}" "${validation_tracked_after}"
    exit "${command_status}"
}
