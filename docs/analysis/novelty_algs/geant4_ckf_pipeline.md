# Running traccc CKF on ColliderML Geant4 Hits — Working Log

**Goal:** Produce realistic pre-AR track candidates from real Geant4 hit topology by running traccc's own Combinatorial Kalman Filter on ColliderML's raw `tracker_hits`. Bypasses the limitation that ColliderML's published `tracks` table is already cleaned (~0.7% shared hits).

**Why this matters:** Synthetic CKF-like injection works but produces a uniform-branching conflict graph that doesn't reproduce FATRAS's algorithmic regime. Natural CKF output on Geant4 hits would give us both real hit physics AND realistic AR conflict structure (long chains, varying branch depths).

Cross-references:
- Inspection that revealed the problem: [`geant4_colliderml_results.md`](geant4_colliderml_results.md)
- FATRAS baseline: [`cpu_multithreaded_results.md`](cpu_multithreaded_results.md)

---

## Plan

1. **Inspect data formats** in detail
   - ColliderML `tracker_hits` schema and surface_id encoding
   - traccc ODD geometry: surface placements + geometry_id list
2. **Surface ID mapping**: derive ColliderML surface_id → traccc geometry_id mapping
   - Try by position: each ColliderML hit's (x,y,z) should match a single traccc surface
   - Cross-check by `(detector, layer_id)` consistency
3. **Convert hits → traccc measurements**: write per-event CSV files in traccc's expected format
4. **Run traccc pipeline** through CKF + dump to ambiguity_input JSON
5. **Benchmark** the natural pre-AR candidates with `traccc_benchmark_resolver`

## Working notes

### Step 1: ColliderML tracker_hits schema (✅)

ColliderML hits have a **4-level hierarchy**: `detector`, `volume_id`, `layer_id`, `surface_id`.
- 214,203 hits per pu200 event (much more than traccc measurements, since these are pre-clustering)
- `surface_id` alone is NOT unique (3216 unique sids but 18,040 unique full triples)
- `volume_id = 16` matches the ODD digi config → same DD4hep volume scheme as ACTS
- Position range: ±1030mm radial, ±3025mm z — realistic ODD dimensions

### Step 2: traccc detray ODD geometry (✅)

- 114 volumes, 19,556 surfaces total
- Each surface has `source` (DD4hep-encoded uint64), `transform.translation` (centre, mm), `mask` (boundaries)
- Volume names are descriptive: `PixelLayer0`, `ShortStripLayer3`, `PixelEndcapP6` etc

### Step 3: Surface mapping attempts (❌ FAILED)

**Attempt A — by-volume position matching**: Match CML (volume_id) to traccc volume index, then position-match surfaces within that volume.
- Result: catastrophic failure. CML vol=17 has 2490 surfaces but traccc vol[17]="PixelBarrel_gap_0" has 4. Volume indices are completely independent between schemes.

**Attempt B — global position matching (centroids)**: Drop volume_id, match each CML surface centroid to globally nearest traccc surface.
- Result: encouraging. Median 5.8mm, 99% within 23mm, 100% within 32mm.
- Cleanly recovered the **subdetector-level mapping**:

  | CML vol | → traccc subdetector |
  |---|---|
  | 16 | PixelEndcapN |
  | 17 | PixelLayer0–3 (barrel) |
  | 18 | PixelEndcapP |
  | 23 | ShortStripEndcapN |
  | 24 | ShortStripLayer (barrel) |
  | 25 | ShortStripEndcapP |
  | 28 | LongStripEndcapN |
  | 29 | LongStripLayer (barrel) |
  | 30 | LongStripEndcapP |

**Attempt C — per-hit position matching (the real test)**:
- Median 25.8 mm distance per hit to nearest traccc surface
- 99.95% of hits land on *some* active surface, but with mm-precision mismatches
- Stratified: pixel barrel hits ~19mm median, strip endcap hits ~45mm median

### Verdict: geometry mismatch blocks the CKF path

The 25mm median per-hit distance is fatal for running traccc's CKF on these hits. ODD silicon modules are ~20–60mm wide, so hits "near" a module but >5mm off the surface will be assigned to the wrong sensor or fail the surface-intersection check entirely.

**Root cause**: ColliderML's DD4hep ODD XML version and traccc's detray-converted ODD use slightly different surface placements. Both are nominally "ODD" but the geometries differ at the mm scale. Reconciling them would require:
1. The exact DD4hep XML ColliderML used (not published in the dataset)
2. Running traccc's detray geometry builder on that exact XML
3. Validating the resulting detray JSON matches ColliderML hit positions

This is a multi-week effort with no guarantee of success, and not a productive use of thesis time.

### Decision: pivot to improved synthetic injection

Abandon the run-CKF-on-ColliderML approach. Instead, replace the simple uniform-branching synthetic injector with a **realistic CKF-tree injector** that mimics the actual conflict-graph topology FATRAS produces naturally:

- **Multi-sibling branching**: each truth track spawns 2–5 candidates (not just 1 alternate), mimicking CKF branching at seed-finding ambiguities
- **Variable branch depth**: random split point per sibling (not fixed 65%), mimicking different CKF branch decisions
- **Cascading siblings**: with some probability, siblings spawn their own siblings → produces long-chain conflicts characteristic of high-pileup FATRAS

Goal: reproduce the JP>greedy regime at high density on top of real Geant4 hit topology, giving us both `(real hit physics) AND (realistic AR algorithmic structure)`.

This is what the user actually needs: realistic conflict graphs on Geant4 hits, not necessarily authentic CKF output.

---

## Tree-injection sweep results (2026-06-30)

Tree injector spawns **2–5 siblings per parent at variable branch points**, with **cascading sub-siblings** up to N levels deep. Designed to produce long-chain conflict structures characteristic of high-pileup CKF output.

| Scenario | params (br/sib/casc/depth) | n_tracks | shared% | Greedy ev/s | JP ev/s | JP/Greedy |
|---|---|---:|---:|---:|---:|---:|
| Raw (no injection) | — | 1395 | 0.7% | 104.8 | 101.8 | 0.97 |
| **Flat light** (legacy 1-sibling) | dup=0.30, frac=0.70 | 1812 | 24.6% | 61.3 | 65.0 | **+6%** ✓ |
| **Tree light** | 0.30 / 2 / 0.20 / 2 | 2202 | 27.8% | 51.3 | 42.3 | **−18%** ✗ |
| **Tree medium** | 0.60 / 3 / 0.40 / 3 | 5473 | 67.0% | 19.1 | 10.7 | **−44%** ✗ |
| **Tree heavy** | 0.80 / 4 / 0.60 / 4 | 21714 | 97.6% | 5.5 | 0.32 | **−94%** ✗✗ |
| Tree stress (1.0/5/0.8/5) | — | 200k+ | 100% | killed | killed | — |

### Critical finding: JP's advantage depends on graph TOPOLOGY, not density

**Same density, different topology = opposite winner**:
- Flat injection at 25% shared → JP wins +6% (matches FATRAS μ=200-ish behaviour)
- Tree injection at 28% shared → greedy wins by 18%

The difference is the *shape* of the conflict graph:
- **Flat injection** creates "stars": one parent ↔ one sibling. Each track has few neighbours. JP's PROPOSE step iterates over a small neighbour list per node.
- **Tree injection** creates dense cliques: siblings-of-siblings share many hits with each other. Each track has many neighbours. JP's PROPOSE cost (O(neighbours) per node) blows up.
- **FATRAS-natural** apparently sits in a "good" regime where JP's batch removal pays off — but we can't reproduce that with simple synthetic injection.

At extreme density (97% shared, 21k tracks), JP becomes **17× slower than greedy** because the PROPOSE/FINALIZE inner loops have to scan thousands of neighbours per track per outer iteration.

### Implications

1. **Synthetic injection can stress AR algorithms but cannot reproduce FATRAS algorithmic dynamics.** The conflict-graph structure that emerges from real CKF branching on real Geant4 hits is fundamentally different from any synthetic recipe we can construct without access to the actual CKF machinery on the matching geometry.

2. **For the thesis: algorithm choice should be informed by conflict-graph structure, not just density.** This is an interesting result in itself — it explains why JP shines on FATRAS at high pile-up but might NOT shine on other simulations or other reconstruction chains. The right algorithm depends on the upstream pipeline.

3. **Recommendation for the practical AR work**: keep using the FATRAS pileup sweep as the natural-stress baseline; use the **flat synthetic injection** on ColliderML Geant4 hits as a "realism check" (it reproduces FATRAS-like JP wins at moderate density); discard the tree injection as an unrealistic worst-case.

### What's NOT pursued further

- Tree injector tuning: even modest cascade depths (depth=2-3) destroy JP performance. No tuning recovers the FATRAS regime.
- Surface-mapping repair: would need the exact DD4hep XML from the ColliderML team. Out of scope for thesis timeline.
- Other Geant4 datasets: nothing open exists with pile-up >200 plus pre-AR candidates.

### Practical recommendation captured in the thesis

The FATRAS μ=0–600 pile-up sweep remains the primary AR benchmark suite. ColliderML pu200 provides:
- A realism check: real Geant4 hits at HL-LHC pile-up have only 6.5% AR rejection vs FATRAS's 22% (FATRAS overestimates AR workload).
- A controlled-ambiguity test: synthetic flat injection on Geant4 hits at 25% shared density reproduces the FATRAS JP-wins-slightly regime, confirming the algorithmic structure observation is portable across simulation backends at matching graph topology.
