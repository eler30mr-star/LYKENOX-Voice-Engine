# NNSVS CPU Runtime Status

## Runtime result

- Isolated env: `tools/nnsvs_env/.venv`
- Python: `3.11.16` x64
- Compiler: MSVC x64 available through `vcvars64.bat`; `cl` reports version `19.44.35223`
- PyTorch CPU: `2.13.0+cpu`
- CUDA: `False`
- NNSVS: `0.1.1`
- `nnmnkwii`: `0.1.3`
- `pyworld`: `0.3.5`
- `pysinsy`: `0.0.5`
- `librosa`: `0.11.0`
- `h5py`: required because `import nnsvs` failed without it

## Official recipe

Recipe checked: `recipes/nit-song070/dev-48k-world` from the official NNSVS repository.
The current upstream checkout does not contain a folder named `dev-test`; the official NIT recipe folders are
`dev-48k-world` and `test-48k-world`.

- Stage `-1` with original HTTP URL: failed because `http://hts.sp.nitech.ac.jp` was unreachable.
- Dataset download by HTTPS: OK from the same official host.
- Stage `0` data preparation: OK.
- Stage `1` feature generation: FAIL.

Exact stage 1 error:

```text
ModuleNotFoundError: No module named 'parallel_wavegan'
```

NNSVS imports `parallel_wavegan.bin.preprocess.logmelfilterbank` from `prepare_features.py` even when the
selected synthesis/vocoder path is WORLD. Because this phase explicitly says not to install ParallelWaveGAN yet,
stage 1 is intentionally stopped here.

## Spanish status

Spanish labels are not started. The runtime problem is separated from Spanish frontend work:

1. Runtime NNSVS: OK.
2. Official data prep: OK.
3. Official feature generation: blocked by `parallel_wavegan` package.
4. Spanish HTS full-context labels: not attempted yet.
5. LYKENOX dataset: not touched.
