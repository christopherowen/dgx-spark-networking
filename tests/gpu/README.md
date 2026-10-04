# GPU tests

`test_oneshot_gpu.py` runs under torchrun on two or more Sparks sharing a
RoCE fabric (see its module docstring for the command). It skips without a
torchrun environment. Run it only inside an owned cluster window with
serving stopped: it constructs runtimes, injects proxy failures and relies
on every rank being present.
