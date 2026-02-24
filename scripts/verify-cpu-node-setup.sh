#!/bin/bash
set -euo pipefail

echo "=== CPU node setup verification (after cache/spack migration to /data) ==="
echo "host=$(hostname)"
echo ""

DATA_WORK="${DATA_WORK:-/data/alice/sbetisor}"
TRACCC_SRC="${TRACCC_SRC:-$DATA_WORK/traccc}"

fail() { echo "FAIL: $*"; exit 1; }
ok()   { echo "OK:   $*"; }

echo "--- 1. Config dirs (.config, .spack) ---"
for d in .config .spack; do
  if [[ -e "$HOME/$d" ]]; then
    if [[ -L "$HOME/$d" ]]; then
      target=$(readlink -f "$HOME/$d")
      ok "$HOME/$d -> $target"
    else
      ok "$HOME/$d exists in home"
    fi
  else
    fail "$HOME/$d missing"
  fi
done
[[ -e "$HOME/.cache" ]] && ok "$HOME/.cache exists" || echo "SKIP: .cache optional"

echo ""
echo "--- 2. data-work symlink ---"
if [[ -L "$HOME/data-work" ]]; then
  target=$(readlink -f "$HOME/data-work")
  ok "~/data-work -> $target"
elif [[ -d "$DATA_WORK" ]]; then
  ok "DATA_WORK=$DATA_WORK exists"
else
  fail "Neither ~/data-work symlink nor $DATA_WORK found"
fi

echo ""
echo "--- 3. Spack ---"
if [[ ! -f "$DATA_WORK/spack/share/spack/setup-env.sh" ]]; then
  fail "Spack not found at $DATA_WORK/spack"
fi
ok "Spack found"

. "$DATA_WORK/spack/share/spack/setup-env.sh"
spack config get config >/dev/null 2>&1 || fail "spack config get failed"
ok "Spack loads and config is readable"

install_root=$(spack config get config | grep -A1 'install_tree:' | grep 'root:' | awk '{print $2}')
if [[ "$install_root" == *"/data/"* ]]; then
  ok "install_tree root on /data: $install_root"
else
  fail "install_tree root not on /data: $install_root"
fi

echo ""
echo "--- 4. Spack env traccc ---"
if spack env list 2>/dev/null | grep -q traccc; then
  ok "traccc env exists"
  spack env activate traccc 2>/dev/null || fail "spack env activate traccc failed"
  ok "traccc env activated"
else
  echo "SKIP: traccc env not found (run spack env create traccc spack.yaml first)"
fi

echo ""
echo "--- 5. traccc build ---"
if [[ -f "$TRACCC_SRC/build/bin/traccc_seq_example" ]]; then
  ok "traccc_seq_example binary exists"
else
  echo "SKIP: traccc_seq_example not built (run cmake + cmake --build from traccc root)"
fi

echo ""
echo "--- 6. Quick traccc run (if built) ---"
GEOM_DIR="$TRACCC_SRC/data/geometries/odd"
INPUT_DIR="$TRACCC_SRC/data/odd/geant4_10muon_10GeV"
if [[ -f "$TRACCC_SRC/build/bin/traccc_seq_example" ]] && [[ -d "$GEOM_DIR" ]] && [[ -d "$INPUT_DIR" ]]; then
  cd "$TRACCC_SRC"
  export TRACCC_TEST_DATA_DIR="$TRACCC_SRC/data"
  if ./build/bin/traccc_seq_example \
    --detector-file=geometries/odd/odd-detray_geometry_detray.json \
    --material-file=geometries/odd/odd-detray_material_detray.json \
    --grid-file=geometries/odd/odd-detray_surface_grids_detray.json \
    --digitization-file=geometries/odd/odd-digi-geometric-config.json \
    --use-acts-geom-source=1 \
    --input-directory=odd/geant4_10muon_10GeV/ \
    --input-events=2 >/dev/null 2>&1; then
    ok "traccc_seq_example ran successfully"
  else
    echo "WARN: traccc_seq_example failed (check TRACCC_TEST_DATA_DIR; may need digitization patch)"
  fi
else
  echo "SKIP: traccc binary or geometry data missing"
fi

echo ""
echo "=== Verification complete ==="
