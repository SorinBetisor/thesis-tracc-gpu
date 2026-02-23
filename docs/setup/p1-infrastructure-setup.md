# Infrastructure & environment validation

## Goal of this phase

Verify that:

* CPU and GPU jobs can be compiled and executed on the Nikhef Stoomboot cluster.
* The CUDA toolchain is usable on interactive GPU nodes.
* A clean and reproducible working directory layout is established for ACTS / traccc development.

---

## Storage

* No writable `/project` area yet.
* Temporary working directory:

  ```
  /data/alice/sbetisor
  ```
* `$HOME` used only for:

  * lightweight thesis repository
  * configuration files
  * small interactive tests

---

## HTCondor CPU sanity check

* Submitted a minimal C program via HTCondor.
* Job executed on worker node (`wn-sate-079`).
* Compilation and execution performed inside job sandbox.
* Output verified: `n=1000000`, `sum=2999997`.

Confirms:

* `el9` image usable
* `gcc` available on worker nodes
* file transfer and logging work correctly

Submit file:

```
compile_and_run.submit
```

---

## Known issues & fixes (CPU)

* **Issue:** GLIBC mismatch when `/bin/bash` was transferred as executable.
* **Cause:** login-node `/bin/bash` incompatible with worker container.
* **Fix:** use a wrapper script (`run.sh`) as the executable, or disable executable transfer.

---

## CUDA interactive sanity check (NVIDIA GPUs)

### Node

* Interactive GPU node:

  ```
  wn-lot-001
  ```
* GPU present:

  ```
  NVIDIA Quadro GV100
  ```

---

### CUDA availability

CUDA toolkits are installed system-wide under `/usr/local`, but `nvcc` is **not on the default PATH**.

Available CUDA versions:

```
/usr/local/cuda-12.3
/usr/local/cuda-12.4
/usr/local/cuda-12.5
/usr/local/cuda-12.8
/usr/local/cuda-12.9
/usr/local/cuda-13.0
```

CUDA **12.5** selected to match the documented library version.

---

### Environment setup (interactive)

```bash
export CUDA_HOME=/usr/local/cuda-12.5
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
hash -r
```

Verification:

```bash
nvcc --version
nvidia-smi
```

---

### Minimal CUDA runtime test

Compile a minimal CUDA program:

```bash
nvcc -O2 hello.cu -o hello
```

Run interactively:

```bash
./hello
```

Expected output:

```
OK, got 123
```

Confirms:

* CUDA compiler (`nvcc`) works
* GPU kernel execution works
* Device ↔ host memory transfers work
* Interactive GPU usage is functional without batch submission

---

## Repository layout decision

* ACTS and traccc cloned into:

  ```
  /data/alice/sbetisor
  ```
* Thesis repository remains lightweight and independent in `$HOME`.
* This layout will be migrated to:

  ```
  /project/<group>/<user>
  ```

  once the project area becomes available.
