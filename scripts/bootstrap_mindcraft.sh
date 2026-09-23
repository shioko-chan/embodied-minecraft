#!/usr/bin/env bash
# Fetch a pinned Mindcraft source revision and only the packages needed by its
# Mineflayer skills and MineCollab techtree validator. The full Mindcraft app has
# optional native UI/LLM dependencies that this focused integration does not use.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

source_dir=.runtime/upstream/mindcraft
deps_dir=.runtime/mindcraft-deps
revision=5f3acc87b479864124173de444f31fa5538f94a6

mkdir -p .runtime/upstream "$deps_dir"
if [[ ! -d "$source_dir/.git" ]]; then
  if [[ -e "$source_dir" ]]; then
    echo "Existing $source_dir is not a Git checkout; inspect it before retrying." >&2
    exit 1
  fi
  git clone --depth 1 https://github.com/mindcraft-bots/mindcraft.git "$source_dir"
  if [[ $(git -C "$source_dir" rev-parse HEAD) != "$revision" ]]; then
    git -C "$source_dir" fetch --depth 1 origin "$revision"
    git -C "$source_dir" checkout --detach "$revision"
  fi
fi
if [[ $(git -C "$source_dir" rev-parse HEAD) != "$revision" ]]; then
  echo "Mindcraft checkout differs from pinned revision $revision." >&2
  exit 1
fi

cp integrations/mindcraft/package.json integrations/mindcraft/package-lock.json "$deps_dir/"
npm ci --prefix "$deps_dir" --no-audit --no-fund
if [[ -L "$source_dir/node_modules" ]]; then
  [[ $(readlink "$source_dir/node_modules") == ../../mindcraft-deps/node_modules ]] || {
    echo "Unexpected Mindcraft node_modules link; inspect it before retrying." >&2
    exit 1
  }
elif [[ -e "$source_dir/node_modules" ]]; then
  echo "Mindcraft checkout already has node_modules; inspect it before retrying." >&2
  exit 1
else
  ln -s ../../mindcraft-deps/node_modules "$source_dir/node_modules"
fi
echo "Mindcraft $revision skills ready in $source_dir"
