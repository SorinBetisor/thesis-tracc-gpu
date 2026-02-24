# P2 – Baseline: build and run traccc (CPU)

**Goal:** Reproducible CPU baseline build of traccc and a minimal runnable example.  
**Outcome:** Build + run commands documented so anyone can reproduce.

---

## Prerequisites (you have these)

- traccc cloned (e.g. under `/data/alice/sbetisor/traccc` or `~/data-work/traccc`).
- Data files fetched: `./traccc/data/traccc_data_get_files.sh` has been run from the traccc repo root (creates `data/` content, `geometries/odd/`, and input samples like `odd/geant4_10muon_10GeV/`).

**Where to run:** Do **not** build on the login node. Use an **interactive CPU node** (e.g. **stbc-i2**: `ssh cpu-i2` or `ssh stbc-i2.nikhef.nl`).

---

## 0. Dependency management with Spack (no sudo)

If you cannot install system packages (e.g. `tbb-devel`), use Spack to install all dependencies into `data-work/`. Run on an **interactive CPU node** (e.g. stbc-i2).

```bash
export DATA_WORK=~/data-work
# Or: export DATA_WORK=/data/alice/sbetisor

cd "$DATA_WORK"
```

**If Spack is not yet cloned:**
```bash
git clone -c feature.manyFiles=true https://github.com/spack/spack.git "$DATA_WORK/spack"
```

**Configure Spack to install under data-work** (avoids home quota):
```bash
mkdir -p "$DATA_WORK/spack/install"
mkdir -p "$DATA_WORK/spack/etc/spack"
cat > "$DATA_WORK/spack/etc/spack/config.yaml" << 'EOF'
config:
  install_tree:
    root: /data/alice/sbetisor/spack/install
    projections:
      all: "{architecture.platform}-{architecture.target}/{name}-{version}-{hash}"
EOF
```
(Replace `/data/alice/sbetisor` with `$DATA_WORK` if your data-work is elsewhere.)

**Create traccc env and install dependencies:**
```bash
. "$DATA_WORK/spack/share/spack/setup-env.sh"
cd "$DATA_WORK/traccc"
spack env create traccc spack.yaml
spack -e traccc concretize -f
spack -e traccc install
```
(First install can take 30+ minutes: builds TBB, Boost, ROOT, ACTS, etc.)

**For later sessions**, activate the env before building:
```bash
. "$DATA_WORK/spack/share/spack/setup-env.sh"
spack env activate traccc
```

---

## 1. Set workspace

Use the path where traccc actually lives (P1 layout: `/data/alice/sbetisor` for large repos).

```bash
# If you use the symlink from commands.md:
# ln -s /data/alice/sbetisor ~/data-work

export TRACCC_SRC=/data/alice/sbetisor/traccc
# Or: export TRACCC_SRC=$DATA_WORK/traccc
cd "$TRACCC_SRC"
```

(If traccc is under your thesis repo’s `external/traccc`, set `TRACCC_SRC` to that path instead.)

---

## 2. Configure (CPU host preset)

From the traccc source directory.

**With Spack** (after `spack env activate traccc`):
```bash
cmake -DTRACCC_USE_SPACK_LIBS=ON -DDETRAY_USE_SYSTEM_NLOHMANN=ON --preset host-fp32 -S . -B build
```
(`-DDETRAY_USE_SYSTEM_NLOHMANN=ON` ensures detray uses Spack’s nlohmann_json 3.12.0, matching Acts and avoiding ABI mismatch.)

**With system dependencies** (TBB, Boost, ROOT installed via dnf):
```bash
cmake --preset host-fp32 -S . -B build
```

- **host-fp32**: CPU build with single precision, ROOT enabled (for examples).
- For double precision use `host-fp64` instead.

If preset list is needed:

```bash
cmake --list-presets
```

---

## 3. Build

```bash
cmake --build build/
```

Binaries end up in `build/bin/`, e.g. `traccc_seq_example`.

---

## 4. Run minimal baseline example

The sequential full-chain example needs geometry files and an input directory. Paths are resolved via `TRACCC_TEST_DATA_DIR` (or build-time data dir). Set it to your traccc `data/` directory so `geometries/odd/` and `odd/` resolve correctly.

**Correct command** (use `=1` for `--use-acts-geom-source`; bare flag misparses the next arg):

```bash
cd "$TRACCC_SRC"
export TRACCC_TEST_DATA_DIR="$TRACCC_SRC/data"

./build/bin/traccc_seq_example \
  --detector-file=geometries/odd/odd-detray_geometry_detray.json \
  --material-file=geometries/odd/odd-detray_material_detray.json \
  --grid-file=geometries/odd/odd-detray_surface_grids_detray.json \
  --digitization-file=geometries/odd/odd-digi-geometric-config.json \
  --use-acts-geom-source=1 \
  --input-directory=odd/geant4_10muon_10GeV/ \
  --input-events=10
```

**What this runs:** `traccc_seq_example` is the **sequential (single-threaded) CPU** full reconstruction chain. It runs entirely on CPU—no GPU. The pipeline: read cells → clusterization → spacepoint formation → seeding → track finding (CKF) → ambiguity resolution → track fitting. Input: 10 events from the ODD geometry, Geant4 10 GeV muon sample.

**Typical output:** Statistics (cells read, measurements, spacepoints, seeds, tracks found/resolved/fitted) and per-stage timings (Read cells, Clusterization, Seeding, Track finding, Track fitting, Wall time). Warnings about duplicate cells or ROOT TH1 replacement are normal and can be ignored for the baseline.

**GPU variants:** For CUDA builds, use `traccc_seq_example_cuda` (and similarly `traccc_throughput_mt_cuda`). The baseline above is CPU-only.

- **input-events=10**: run 10 events (minimal baseline).
- **--use-acts-geom-source=1**: required; bare `--use-acts-geom-source` causes `invalid_bool_value` (next arg misparsed).
- If the data script created a different path under `data/`, adjust `TRACCC_TEST_DATA_DIR` and `--input-directory` to match.
- **Acts/Spack compatibility:** If `traccc_seq_example` segfaults in `Acts::from_json` when reading the digitization config, apply the workaround in `traccc/io/src/json/read_digitization_config.cpp` (manual BinUtility parsing instead of `Acts::from_json`).

Success = program runs to completion without crash; you may see logging and event counts.

---

## 5. Optional: throughput example

Same geometry, more events and threads (still CPU):

```bash
./build/bin/traccc_throughput_mt \
  --detector-file=geometries/odd/odd-detray_geometry_detray.json \
  --material-file=geometries/odd/odd-detray_material_detray.json \
  --grid-file=geometries/odd/odd-detray_surface_grids_detray.json \
  --digitization-file=geometries/odd/odd-digi-geometric-config.json \
  --use-acts-geom-source=1 \
  --input-directory=odd/geant4_10muon_10GeV/ \
  --input-events=10 \
  --processed-events=1000 \
  --threads=1
```

---

## 6. Reproducibility checklist

- [ ] Build done on **interactive CPU node** (e.g. stbc-i2), not login node.
- [ ] `TRACCC_SRC` (or path) and preset (`host-fp32` / `host-fp64`) recorded.
- [ ] Exact `cmake --preset` and `cmake --build` commands written down (see sections 2–3).
- [ ] Exact `traccc_seq_example` (and optional `traccc_throughput_mt`) command with all arguments recorded (see section 4–5).
- [ ] Data source: `traccc_data_get_files.sh` run once from traccc repo root; geometry and input paths documented.

---

## Troubleshooting

- **CMake preset not found:** Run from `TRACCC_SRC` and use `-S . -B build`. Run `cmake --list-presets` to see presets.
- **Missing TBB (TBBConfig.cmake / tbb-config.cmake):** The host presets set `TRACCC_USE_SYSTEM_TBB=TRUE`. With sudo: `dnf install tbb-devel`. Without sudo: use Spack (see **section 0**).
- **Acts::from_json / Acts::to_json undefined reference:** nlohmann_json ABI mismatch: detray fetches 3.11.x, Acts (Spack) uses 3.12.0. Reconfigure with `-DDETRAY_USE_SYSTEM_NLOHMANN=ON` so detray uses Spack’s nlohmann_json.
- **Acts::Core target not found:** Spack-built Acts exports `ActsCore`, not `Acts::Core`. Add aliases to the Acts config: append to `$ACTSPREFIX/lib64/cmake/Acts/ActsConfig.cmake` (where `$ACTSPREFIX` is the Acts install path in the Spack view) after the `endforeach()`:
  ```cmake
  if(TARGET ActsCore AND NOT TARGET Acts::Core)
    add_library(Acts::Core ALIAS ActsCore)
  endif()
  if(TARGET ActsPluginJson AND NOT TARGET Acts::PluginJson)
    add_library(Acts::PluginJson ALIAS ActsPluginJson)
  endif()
  ```
- **`invalid_bool_value` for `--use-acts-geom-source`:** Use `--use-acts-geom-source=1` (or `=on`), not bare `--use-acts-geom-source`.
- **Missing geometry or input files:** Ensure you ran `./traccc/data/traccc_data_get_files.sh` from the traccc repo root and that you run the binary from that same root (or fix paths in `--detector-file`, `--input-directory`, etc.).
- **Dependencies (Boost, ROOT):** On Stoomboot, use system modules or a Spack env if needed; see traccc README “Dependency management with Spack” if you hit missing libraries.

Once this runs reliably, P2 baseline is done. Next: document the exact commands in this file (or a “Commands” subsection) and move on to CUDA/GPU when ready.
