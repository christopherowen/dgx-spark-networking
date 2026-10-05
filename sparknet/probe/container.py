"""The bounded Docker command that runs the collective probe on one rank. A plan, never executed here."""

from __future__ import annotations

PROBE_CONTAINER = "sparknet-collective-probe"
# Where a vLLM nightly image installs the package; ``package_source`` is mounted over it.
DEFAULT_PACKAGE_TARGET = "/usr/local/lib/python3.12/dist-packages/sparknet"


def docker_probe_command(
    *,
    image: str,
    environment: dict[str, str],
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    transport: str,
    probe_source: str | None = None,
    package_source: str | None = None,
    package_target: str = DEFAULT_PACKAGE_TARGET,
    memory: str = "12g",
    timeout_seconds: int = 600,
    extra_args: tuple[str, ...] = (),
) -> list[str]:
    """One rank's probe container: separate IPC namespace, bounded memory and lifetime.

    The 12 GiB limit covers two NCCL communicators plus cold one-shot
    compilation. By default the probe is the one installed in the image
    (``python3 -m sparknet.probe.collectives``); ``probe_source`` mounts a
    newer ``collectives.py`` from the host instead, and ``package_source``
    mounts a whole ``sparknet`` package directory from the host over the
    image's, so a checkout can be probed through a released image (the proxy
    is rebuilt from the mounted source on first use).
    """
    env = dict(environment)
    env.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,GRAPH,NET")
    command = ["docker", "run", "--rm", f"--name={PROBE_CONTAINER}",
               "--network=host", "--gpus=all", f"--memory={memory}", f"--memory-swap={memory}",
               "--device=/dev/infiniband:/dev/infiniband:rwm", "--cap-add=IPC_LOCK",
               "--ulimit=memlock=-1:-1", "--shm-size=256m",
               "--entrypoint=/usr/bin/timeout"]
    for key, value in sorted(env.items()):
        command.extend(["--env", f"{key}={value}"])
    if package_source:
        command.extend(["--volume", f"{package_source}:{package_target}:ro"])
    if probe_source:
        command.extend(["--volume", f"{probe_source}:/probe.py:ro"])
        program = ["python3", "/probe.py"]
    else:
        program = ["python3", "-m", "sparknet.probe.collectives"]
    command.extend([image, "--signal=TERM", "--kill-after=10s", f"{timeout_seconds}s",
                    *program, "--rank", str(rank), "--world-size", str(world_size),
                    "--master-addr", master_addr, "--master-port", str(master_port),
                    "--transport", transport, *extra_args])
    return command


__all__ = ["DEFAULT_PACKAGE_TARGET", "PROBE_CONTAINER", "docker_probe_command"]
