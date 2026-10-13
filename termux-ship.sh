#!/data/data/com.termux/files/usr/bin/bash
# Termux helper: push GameLens to GitHub, wait for the Actions build, download the APK.
# Usage:
#   bash termux-ship.sh setup            # one-time: install git + gh, log in
#   bash termux-ship.sh push [repo-name] # create repo (first time), commit, push
#   bash termux-ship.sh apk              # wait for the latest build and download the APK
set -euo pipefail

cmd="${1:-}"

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
    [ -d .git ] || { git init -b main; }
    git add -A
    git commit -m "Update GameLens" || echo "Nothing new to commit."
    if ! git remote get-url origin >/dev/null 2>&1; then
      gh repo create "$name" --private --source=. --remote=origin --push
    else
      git push -u origin main
    fi
    echo "Pushed. Build started: run 'bash termux-ship.sh apk' to fetch the APK."
    ;;

  apk)
    sleep 5
    run_id="$(gh run list --workflow=build-apk.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
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
