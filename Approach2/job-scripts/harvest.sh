#!/bin/bash
# Commit and push whatever results and audit reports a finished job left behind.
#
# Launchers stopped auto-committing on purpose: a compute node committing into a
# shared clone is what killed queued jobs. The cost is that every harvest leaves
# the tree dirty, and S1 refuses to prepare a submission from a dirty tree — so
# after each job this has to happen by hand. One line instead of four:
#
#   bash Approach2/job-scripts/harvest.sh
#   bash Approach2/job-scripts/harvest.sh "results: the thing I actually ran"
#
# It only ever touches Approach2/results and Approach2/audits, and refuses when
# anything else is dirty, so it can never sweep an unfinished code change into a
# commit that a submission then freezes.
set -uo pipefail

MESSAGE="${1:-}"
# git aborts on a pathspec that matches nothing, so only existing directories go
# in; a clone without audits/ yet must still be able to harvest results.
PATHS=()
for d in "Approach2/results" "Approach2/audits"; do
  [ -d "$d" ] && PATHS+=("$d")
done

[ -d Approach2 ] || { echo "run this from the repo root"; exit 1; }
[ ${#PATHS[@]} -gt 0 ] || { echo "neither Approach2/results nor Approach2/audits exists"; exit 1; }

OTHER=$(git status --porcelain -- . ':!Approach2/results' ':!Approach2/audits')
if [ -n "$OTHER" ]; then
  echo "refusing: something outside results/ and audits/ is uncommitted."
  echo "$OTHER"
  echo "Commit or stash that first; a submission freezes whatever commit this makes."
  exit 1
fi

if [ -z "$(git status --porcelain -- "${PATHS[@]}")" ]; then
  echo "nothing new in results/ or audits/"
  exit 0
fi

git add -- "${PATHS[@]}"

if [ -z "$MESSAGE" ]; then
  # The job id lives inside each summary the evaluators write, so the message can
  # name the runs it is committing instead of a date nobody can trace back.
  MESSAGE=$(git diff --cached --name-only -- "${PATHS[@]}" | python3 -c '
import json, os, sys
files = [line.strip() for line in sys.stdin if line.strip()]
jobs = set()
for name in files:
    if name.endswith(".summary.json") and os.path.exists(name):
        try:
            job = json.load(open(name)).get("slurm_job_id")
        except Exception:
            continue
        if job:
            jobs.add(str(job))
joined = ", ".join(sorted(jobs))
tail = " (jobs " + joined + ")" if joined else ""
print("results: " + str(len(files)) + " files harvested" + tail)')
fi

git commit -q -m "$MESSAGE" || exit 1
echo "committed: $MESSAGE"
git pull -q --rebase || { echo "pull --rebase failed; resolve, then push by hand"; exit 1; }
git push -q && echo "pushed"
