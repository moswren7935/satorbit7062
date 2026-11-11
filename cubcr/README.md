# cuBCR

Hardware used: 2 × Intel Xeon Gold 6326 (32 physical cores total), 1 TiB RAM, and NVIDIA A100 80 GB PCIe.

Environment: Ubuntu 22.04, Python 3.10, GCC 11.4/OpenMP, CUDA 12.5. Requires AVX-512F, GNU Make, Bash, curl, tar, and coreutils. The default GPU target is `sm_80`.

## Setup

Run from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export NVCC=nvcc
make -C csrc all
python smoke_test.py --engine both
```

Install `python3.10-venv` if needed. Set `NVCC` to your CUDA 12.5 compiler path if it is not on `PATH`.

## Reproduce

Install [PRISM-games](https://www.prismmodelchecker.org/games/installation.php), Java 11+, and [Storm](https://www.stormchecker.org/documentation/obtain-storm/build.html). Set the executable paths below; Java is found through `JAVA_HOME` or `PATH`.

```bash
export CUBCR_PRISM_BIN=/path/to/prism-games/bin/prism
export CUBCR_STORM_BIN=/path/to/storm
python experiments/rerun_revised.py --stage preparation primary engine tables decision overcooked validation E1 E2 E3 E4 E5 E7 E9 report
```

The drivers generate TicketBench and synthetic games. The installed `overcooked-ai` package supplies human training data; the drivers generate its models. The `preparation` and `validation` stages download comparison implementations and PRISM benchmarks. Installation and downloads require internet access.

Measurements go to `results/` and `rebuttal/results/`; tables, figures, and numerical summaries go to `paper/revised/`. Use `--help` to select stages or `--dry-run` to list commands. Full reproduction requires long-running experiments; timing results depend on hardware and system load.
