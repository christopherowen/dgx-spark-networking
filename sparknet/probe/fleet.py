"""Run the collective probe on every node of a map at once, over ssh, and keep the receipts.

One ssh session per node runs that rank's bounded probe container
(``sparknet.probe.container``); the ranks rendezvous on the head node's
management address. Each node's output goes to ``<out>/<node>.log`` and the
rank's JSON result, when it printed one, to ``<out>/<node>.json``. The runner
starts and waits; it never stops or starts anything else on a node, and it
is meant to run inside whatever cluster window a site uses, with serving
stopped.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from sparknet.probe.container import docker_probe_command
from sparknet.topology import nodes as topology

DEFAULT_SSH = ("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15")


@dataclass
class RankPlan:
    name: str
    rank: int
    target: str
    command: list[str]


@dataclass
class RankOutcome:
    name: str
    rank: int
    returncode: int | None
    log: Path
    result: dict | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and self.result is not None and bool(self.result.get("passed")) and not self.problems


def plan(
    nodes: dict,
    environments: dict[str, dict[str, str]],
    *,
    transport: str,
    image: str,
    port: int,
    probe_args: tuple[str, ...] = (),
    probe_source: str | None = None,
    package_source: str | None = None,
    extra_env: dict[str, str] | None = None,
    ssh_user: str | None = None,
    target_field: str = "name",
) -> list[RankPlan]:
    """One plan per node: the ssh target and the probe container command for its rank."""
    head = topology.head_node(nodes)
    user = ssh_user or nodes.get("ssh_user")
    plans = []
    for node in sorted(nodes["nodes"], key=lambda n: n["rank"]):
        env = dict(environments[node["name"]])
        env.update(extra_env or {})
        command = docker_probe_command(
            image=image, environment=env, rank=node["rank"], world_size=len(nodes["nodes"]),
            master_addr=head["management_ip"], master_port=port, transport=transport,
            probe_source=probe_source, package_source=package_source, extra_args=tuple(probe_args),
        )
        host = node[target_field] if target_field != "name" else node["name"]
        plans.append(RankPlan(node["name"], node["rank"], f"{user}@{host}" if user else host, command))
    return plans


def bootstrap_warnings(nodes: dict, environments: dict[str, dict[str, str]],
                       extra_env: dict[str, str] | None = None) -> list[str]:
    """Ranks whose Gloo bootstrap would bind to the address their hostname resolves to.

    Without ``GLOO_SOCKET_IFNAME`` (rendered from the map's ``management_interface``),
    Gloo binds to the hostname's address, which many hosts map to 127.0.0.1; the
    peers then cannot connect and the probe times out in its process group setup.
    """
    warnings = []
    for node in sorted(nodes["nodes"], key=lambda n: n["rank"]):
        env = dict(environments.get(node["name"], {}))
        env.update(extra_env or {})
        if "GLOO_SOCKET_IFNAME" not in env:
            warnings.append(
                f"{node['name']}: no GLOO_SOCKET_IFNAME (the map has no management_interface); the probe's Gloo "
                "bootstrap binds to the hostname's address, which is loopback on many hosts; add "
                "management_interface to the map or pass --env GLOO_SOCKET_IFNAME=<interface>"
            )
    return warnings


def ssh_command(item: RankPlan, ssh: tuple[str, ...] = DEFAULT_SSH) -> list[str]:
    return [*ssh, item.target, shlex.join(item.command)]


def _result_from_log(log: Path) -> dict | None:
    """The rank's final JSON result: the last line that parses and carries ``passed``."""
    result = None
    for line in log.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict) and "passed" in parsed:
            result = parsed
    return result


def run(plans: list[RankPlan], out: str | Path, *, ssh: tuple[str, ...] = DEFAULT_SSH, timeout: float = 900.0) -> list[RankOutcome]:
    """Start every rank's ssh session together, wait for all of them, and collect the results."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = []
    for item in plans:
        log = out / f"{item.name}.log"
        handle = log.open("wb")
        handle.write((shlex.join(ssh_command(item, ssh)) + "\n").encode())
        handle.flush()
        process = subprocess.Popen(ssh_command(item, ssh), stdout=handle, stderr=subprocess.STDOUT)
        started.append((item, log, handle, process))
    deadline = time.monotonic() + timeout
    outcomes = []
    for item, log, handle, process in started:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        handle.close()
        outcome = RankOutcome(item.name, item.rank, process.returncode, log)
        outcome.result = _result_from_log(log)
        if outcome.result is not None:
            (out / f"{item.name}.json").write_text(json.dumps(outcome.result, indent=2) + "\n")
            if outcome.result.get("rank") != item.rank:
                outcome.problems.append(f"result names rank {outcome.result.get('rank')}, expected {item.rank}")
        elif process.returncode == 0:
            outcome.problems.append("exited 0 without a JSON result")
        if process.returncode is None or process.returncode != 0:
            outcome.problems.append(f"exit status {process.returncode}")
        outcomes.append(outcome)
    return outcomes


__all__ = ["DEFAULT_SSH", "RankOutcome", "RankPlan", "plan", "run", "ssh_command"]
