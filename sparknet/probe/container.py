"""The bounded Docker command that runs the collective probe on one rank. A plan, never executed here."""

from __future__ import annotations

PROBE_CONTAINER = "sparknet-collective-probe"


def docker_probe_command(
    *,
    image: str,
    environment: dict[str, str],
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    transport: str,
    probe_source: str,
    memory: str = "12g",
    timeout_seconds: int = 600,
    extra_args: tuple[str, ...] = (),
) -> list[str]:
    """One rank's probe container: separate IPC namespace, bounded memory and lifetime.

    The 12 GiB limit covers two NCCL communicators plus cold one-shot
    compilation. ``probe_source`` is the path of ``sparknet/probe/collectives.py``
    on the host (or an installed sparknet package directory) mounted read-only.
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
    command.extend(["--volume", f"{probe_source}:/probe.py:ro", image,
                    "--signal=TERM", "--kill-after=10s", f"{timeout_seconds}s",
                    "python3", "/probe.py", "--rank", str(rank), "--world-size", str(world_size),
                    "--master-addr", master_addr, "--master-port", str(master_port),
                    "--transport", transport, *extra_args])
    return command


__all__ = ["PROBE_CONTAINER", "docker_probe_command"]
