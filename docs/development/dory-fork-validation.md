# Dory fork validation

This fork branch starts at Dory PR #30306 commit `57b70d02f084d7640d17260a179ef73616d9a438`.

The first commit changes six scale tensor roles to probe their BF16-to-F32 cast during buffer selection and adds the tiny BF16 fixture to the existing architecture test. The second commit rejects unsupported Dory tensor splitting before allocation, avoiding the existing Meta split divide-by-zero. Neither changes CUDA kernels or model arithmetic.

The third commit is fork-only CI infrastructure. It is not part of the proposed runtime patch. The workflow runs Linux Release, Linux ASan/UBSan, Windows static Release, and a CUDA compile check using the upstream Ubuntu CUDA job's toolkit, architecture, CCCL pin, and warning settings. This is a focused validation matrix, not the complete upstream CI suite. Logs are uploaded as artifacts and identify the tested commit.

The CI follow-up keeps fatal warnings enabled for Linux and CUDA. Windows uses the upstream CPU workflow's default of nonfatal warnings: this base contains GCC unroll pragmas that MSVC diagnoses as C4068. The CUDA container installs Git before checkout so checkout uses Git rather than the archive fallback and can record the tested SHA. These workflow changes do not alter the reviewed C++ source.

The first combined sanitizer run passed the Dory BF16 fixture and quantization test, then stopped in unchanged `tests/test-gguf.cpp:160`: its `get_kv_types` helper constructs the sentinel `gguf_type(-1)`, which UBSan rejects. The next run passed all nine Dory configurations with ASan+UBSan and GGUF with ASan, then stopped in unchanged `tests/test-backend-ops.cpp:11471`: case generation combines resize flags into `ggml_scale_mode` before the CPY filter runs, which UBSan also rejects. The final workflow retains ASan+UBSan for Dory and quantization, and builds/runs GGUF and cast tests separately under ASan. Both original failure logs are preserved. No sanitizer finding in the runtime was suppressed. The CUDA container also trusts its exact checkout path for Git's ownership check.

The CPU fixture validates BF16 loading, the 19-scale fixture, and mixed-batch inference. It cannot detect CUDA placement regressions: CPU fallback is valid for a CPU device. The hosted CUDA job compiles the backend and tests but does not execute GPU tests. Neither hosted result replaces the GPU regression control.

Local validation of the same four source files on one RTX 5070 Ti passed 126 Dory CPU/CUDA cases across 1-4 emulated CUDA devices. These are not multiple physical GPUs. The placement test rejected forced CUDA pinned-host storage and a fixture with one missing BF16 scale; restoring the sources passed. An earlier Linux CUDA baseline/fixed comparison failed placement before the correction and passed afterward. That earlier Linux build predates the final assertion tightening and unsupported-mode guard. The production CLI now rejects Dory tensor splitting cleanly.

Claude Opus 5.5 reviewed the final source and recorded no blockers. It identified a remaining hypothetical test limitation for backends whose operation-support checks depend on the source's existing buffer; CUDA's placement negative control does fail as intended. Dory model-save round trips remain unsupported and are skipped by the existing test harness.

The separate Dory partial-offload issue remains open: with two loops, `-ngl 4` and `-ngl 40` assigned 0/72 physical repeating layers to CUDA, while `-ngl 999` assigned 72/72. Only the output head was offloaded at the smaller settings. This branch does not implement partial-offload or tensor-parallel mapping.

AI assistance: Codex implemented and tested these changes under the owner's direction; Claude Opus 5.5 provided independent read-only review.
