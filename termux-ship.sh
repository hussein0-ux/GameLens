#!/data/data/com.termux/files/usr/bin/bash
# Termux helper: push GameLens to GitHub, wait for the Actions build, download the APK.
# Safe to re-run: works whether the GitHub repo is new, already exists, or already has commits.
# Usage:
#   bash termux-ship.sh setup            # one-time: install git + gh, log in
#   bash termux-ship.sh push [repo-name] # create/link repo, commit, sync with GitHub, push
#   bash termux-ship.sh apk              # wait for the build of your latest push and download the APK
set -euo pipefail

cmd="${1:-}"

# Make sure the "origin" remote points at <you>/<name> on GitHub, creating the repo only if missing.
link_repo() {
  local name="$1" owner url
  owner="$(gh api user --jq .login)"
  url="https://github.com/$owner/$name.git"

  if gh repo view "$owner/$name" >/dev/null 2>&1; then
    echo "Using existing GitHub repo $owner/$name"
  else
    echo "Creating private GitHub repo $owner/$name"
    gh repo create "$owner/$name" --private
  fi

  if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$url"
  else
    git remote add origin "$url"
  fi
}

case "$cmd" in
  setup)
    pkg update -y
    pkg install -y git gh termux-api
    gh auth login
    if ! git config --global user.name >/dev/null 2>&1; then
      read -rp "Git user.name: " n
      git config --global user.name "${n:-GameLens Dev}"
    fi
    if ! git config --global user.email >/dev/null 2>&1; then
      read -rp "Git user.email: " e
      git config --global user.email "${e:-dev@example.com}"
    fi
    ;;

  push)
    name="${2:-GameLens}"

    [ -d .git ] || git init -b main
    git branch -M main
    git config core.fileMode false   # shared storage has no exec bits; avoid phantom changes
    gh auth setup-git
    link_repo "$name"

    git add -A
    git commit -m "Update GameLens" || echo "No new local changes to commit."

    # If GitHub already has history that this folder doesn't contain (earlier push, unrelated
    # history, etc.), keep that history and stack the files in this folder on top of it.
    # The files here always win, which is what you want when you unzip a newer version over the old one.
    if git fetch origin main >/dev/null 2>&1; then
      if ! git merge-base --is-ancestor origin/main HEAD; then
        echo "GitHub already has other history; putting your current files on top of it..."
        git reset --soft origin/main
        git commit -m "Update GameLens" || echo "GitHub already matches your files."
      fi
    fi

    git push -u origin main
    echo "Pushed. Build started: run 'bash termux-ship.sh apk' to fetch the APK."
    ;;

  apk)
    sha="$(git rev-parse HEAD 2>/dev/null || true)"
    run_id=""
    echo "Looking for the build of your latest push..."
    for _ in $(seq 1 12); do
      if [ -n "$sha" ]; then
        run_id="$(gh run list --workflow=build-apk.yml --commit "$sha" --limit 1 --json databaseId --jq '.[0].databaseId // empty' 2>/dev/null || true)"
      fi
      [ -n "$run_id" ] && break
      sleep 5
    done
    if [ -z "$run_id" ]; then
      echo "No run found for this exact commit; using the most recent build."
      run_id="$(gh run list --workflow=build-apk.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
    fi

    echo "Watching run $run_id ..."
    gh run watch "$run_id" --exit-status
    rm -rf apk-out
    gh run download "$run_id" --name GameLens-debug-apk --dir apk-out
    mkdir -p ~/storage/downloads 2>/dev/null || true
    if [ -d ~/storage/downloads ]; then
      cp apk-out/app-debug.apk ~/storage/downloads/GameLens.apk
      echo "Saved to Downloads/GameLens.apk. Open it from the Files app to install."
      command -v termux-open >/dev/null && termux-open ~/storage/downloads/GameLens.apk || true
    else
      echo "APK at $(pwd)/apk-out/app-debug.apk. Run 'termux-setup-storage' then re-run to copy to Downloads."
    fi
    ;;

  *)
    echo "Usage: bash termux-ship.sh {setup|push [repo-name]|apk}"
    exit 1
    ;;
esac
