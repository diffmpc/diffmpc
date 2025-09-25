# Benchmarks for diffmpc

## Installation
Separate environments are used for each of the solvers to avoid dependency conflicts. 
Installation instructions for each environment are below.

### diffmpc
```bash
python -m venv .diffmpc-venv && .diffmpc-venv/bin/activate
# cd to diffmpc root directory
python -m pip install -e .
```

## mpc.pytorch
```bash
python -m venv .mpcpt-venv && .mpcpt-venv/bin/activate
cd mpc.pytorch
python -m pip install -e .
pip install jax # jax is also required to run the mpc.pytorch benchmark
```

## theseus
```bash
python -m venv .theseus-venv && .theseus-venv/bin/activate
python -m pip install theseus-ai
pip install torch jax
```

## trajax
This environment uses a docker container for specific package versions.
```bash
cd trajax_env
./build.sh
./run.sh
```

# Running Benchmarks
## Reinforcement Learning 
Reinforcement learning benchmarks are in `benchmarks/reinforcement-learning`. 

Activate the virtual environemnt `source [solver]-venv/bin/activate` where `[solver]` corresponds to the solver evaluate. Then, run `python benchmark_<solver_name>.py`. Timing results are written to `reinforcement-learning/timing_results` and can be printed using `print_timing_results.py`.

## Imitation Learning
Reinforcement learning benchmarks are in `benchmarks/imitation-learning`. 

Activate the virtual environemnt `source [solver]-venv/bin/activate` where `[solver]` corresponds to the solver evaluate. 

### Execution on the CPU vs on the GPU
Pytorch solver benchmarks (`theseus` and `mpc.pytorch`) expose a command line flag for `--device`

Jax solvers (`trajax` and `diffmpc`) are forced to use the CPU by setting the `JAX_PLATFORMS` environment variable as `export JAX_PLATFORMS="cpu"`.
