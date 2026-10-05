"""``sparknet``: topology, NCCL profile, policy and probe tooling for DGX Spark fabrics.

Every command here runs without torch. The collective probe itself
(``sparknet probe collectives``) imports torch on the node it runs on.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from sparknet import __version__
from sparknet.nccl import patchset, profiles as nccl_profiles
from sparknet.policy import CollectivePolicy, policy_for_profile
from sparknet.probe.container import docker_probe_command
from sparknet.probe.doctor import local_problems
from sparknet.probe.gpudirect import gpudirect_report
from sparknet.topology import discover, nodes as topology, render

EXAMPLES = Path(__file__).parent / "topology" / "examples"


def _print_env(env: dict[str, str], as_json: bool) -> None:
    if as_json:
        print(json.dumps(env, indent=2, sort_keys=True))
    else:
        for key in sorted(env):
            print(f"{key}={shlex.quote(env[key])}")


def _report(errors: list[str], what: str) -> int:
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"{what}: ok")
    return 0


def rendered_environment(nodes: dict, node: dict, *, transport: str, profile: str | None) -> dict[str, str]:
    base = nccl_profiles.environment(profile) if profile else {}
    return render.node_environment(nodes, node, transport=transport, base=base)


def cmd_topology_validate(args) -> int:
    nodes = topology.load(args.nodes)
    return _report(topology.problems(nodes, args.transport, mesh_paths=args.mesh_paths), f"{args.nodes} ({args.transport})")


def _profile_errors(args, nodes: dict) -> list[str]:
    errors = topology.problems(nodes, args.transport, mesh_paths=getattr(args, "mesh_paths", 2))
    if not errors and args.profile:
        errors = nccl_profiles.profile_problems(args.profile, args.transport, len(nodes["nodes"]))
        if errors:
            fitting = nccl_profiles.profiles_for(args.transport, len(nodes["nodes"]))
            errors.append("profiles for this map: " + (", ".join(fitting) if fitting else "none"))
    return errors


def cmd_topology_render(args) -> int:
    nodes = topology.load(args.nodes)
    errors = _profile_errors(args, nodes)
    if errors:
        return _report(errors, args.nodes)
    node = topology.node_by_name(nodes, args.node)
    env = rendered_environment(nodes, node, transport=args.transport, profile=args.profile)
    errors = render.environment_problems(env, args.transport, len(nodes["nodes"]))
    if args.profile:
        errors += nccl_profiles.problems(env, node_count=len(nodes["nodes"]))
    if errors:
        return _report(errors, "rendered environment")
    _print_env(env, args.json)
    return 0


def cmd_topology_example(args) -> int:
    path = EXAMPLES / f"{args.name}.json"
    if not path.exists():
        print(f"error: no example {args.name!r}; choose {', '.join(p.stem for p in sorted(EXAMPLES.glob('*.json')))}", file=sys.stderr)
        return 1
    print(path.read_text(), end="")
    return 0


def cmd_topology_discover(args) -> int:
    if args.fabric == "switched":
        return _discover_switched(args)
    data = {host: discover.collect_lldp(host, ssh_user=args.ssh_user) for host in args.hosts}
    links, problems = discover.resolve_links(data)
    numbers = {h: discover.node_number(h) for h in args.hosts}
    for (host, iface), (peer, peer_iface) in sorted(links.items()):
        path = discover.INTERFACES[iface][1]
        print(f"{host} {iface:<14} <-> {peer} {peer_iface:<14}  {discover.cable_subnet(numbers[host], numbers[peer], path)}")
    for problem in problems:
        print("WARNING:", problem, file=sys.stderr)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for host in args.hosts:
            (out / host).mkdir(exist_ok=True)
            (out / host / "40-cx7.yaml").write_text(discover.netplan_yaml(host, links, numbers))
        if args.management_ip:
            ips = dict(item.split("=", 1) for item in args.management_ip)
            document = discover.generate_nodes(links, args.hosts, management_ips=ips, ssh_user=args.ssh_user or "spark",
                                               management_interface=args.management_interface)
            discover.write_json(out / "nodes.json", document)
            errors = topology.problems(document, "nccl-ring" if len(args.hosts) == 4 else "oneshot-direct")
            if errors:
                return _report(errors, "generated node map")
        print(f"wrote {out}/<host>/40-cx7.yaml" + (" and nodes.json" if args.management_ip else ""))
    return 1 if problems else 0


def _discover_switched(args) -> int:
    rails = {host: discover.collect_rails(host, ssh_user=args.ssh_user, gid_index=args.gid_index) for host in args.hosts}
    for host in args.hosts:
        for hca, entry in sorted(rails[host].items()):
            print(f"{host} {hca:<14} {entry['state']:<10} {discover.gid_ipv4(entry['gid']) or entry['gid']:<16} {entry['netdev']} mtu {entry['mtu']}")
    if not args.management_ip:
        print("pass --management-ip HOST=IP for every host to generate nodes.json", file=sys.stderr)
        return 0
    ips = dict(item.split("=", 1) for item in args.management_ip)
    document, problems = discover.generate_switched_nodes(rails, args.hosts, management_ips=ips, ssh_user=args.ssh_user or "spark",
                                                          management_interface=args.management_interface, gid_index=args.gid_index,
                                                          traffic_class=args.traffic_class)
    for problem in problems:
        print("WARNING:", problem, file=sys.stderr)
    errors = topology.problems(document, "oneshot-switched")
    if errors:
        return _report(errors, "generated switched node map")
    out = Path(args.out or ".")
    out.mkdir(parents=True, exist_ok=True)
    discover.write_json(out / "nodes.json", document)
    print(f"wrote {out}/nodes.json")
    return 1 if problems else 0


def cmd_topology_inventory(args) -> int:
    print(json.dumps(discover.local_inventory(gid_index=args.gid_index), indent=2))
    return 0


def cmd_nccl_env(args) -> int:
    env = nccl_profiles.environment(args.profile)
    _print_env(env, args.json)
    return 0


def cmd_nccl_validate(args) -> int:
    env = nccl_profiles.environment(args.profile)
    if args.env_file:
        for line in Path(args.env_file).read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()
    counts = nccl_profiles.profile(args.profile)["node_counts"]
    count = args.nodes if args.nodes else counts[0]
    errors = nccl_profiles.profile_problems(args.profile, nccl_profiles.profile(args.profile)["transport"], count)
    errors += nccl_profiles.problems(env, node_count=count, patched_nccl=not args.unpatched)
    return _report(errors, f"profile {args.profile} ({count} nodes)")


def cmd_nccl_profiles(args) -> int:
    for name, entry in nccl_profiles.PROFILES.items():
        counts = entry["node_counts"]
        nodes = f"{counts[0]}-{counts[-1]}" if len(counts) > 2 else " or ".join(str(c) for c in counts)
        print(f"{name}: {nodes} nodes, {entry['transport']}; {entry['status']}")
        print(f"  patches: {', '.join(nccl_profiles.required_patches(nccl_profiles.environment(name)))}")
    return 0


def cmd_nccl_patches(args) -> int:
    if args.export:
        written = patchset.export(args.export)
        print(f"wrote {len(written)} files to {Path(args.export)}: " + ", ".join(path.name for path in written))
        return 0
    print(patchset.series_text(), end="")
    return 0


def cmd_policy_show(args) -> int:
    if args.profile:
        policy = policy_for_profile(args.profile)
        if policy is None:
            print(f"{args.profile}: NCCL carries every collective (no one-shot runtime)")
            return 0
    else:
        try:
            policy = CollectivePolicy.from_environment(dict(os.environ))
        except ValueError as error:
            return _report([f"{error} (set SPARKNET_ROCE_* in the environment or pass --profile)"], "policy")
    print(json.dumps({**policy.__dict__, "reduce_scatter": policy.reduce_scatter_backend(),
                      "environment": policy.environment()}, indent=2))
    return 0


def cmd_probe_doctor(args) -> int:
    nodes = topology.load(args.nodes)
    return _report(local_problems(nodes, args.node, transport=args.transport), f"{args.node} fabric")


def cmd_probe_gpudirect(args) -> int:
    report = gpudirect_report(gpunetio_root=args.gpunetio_dir)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        for stage, state in report["stages"].items():
            print(f"{stage}: {'ready' if state['ready'] else 'not ready'}")
            for need in state["needs"]:
                print(f"  needs: {need}")
        print(f"host proxy: {'available' if report['host_proxy'].get('available') else 'unavailable'}")
    return 0


def cmd_probe_render_command(args) -> int:
    nodes = topology.load(args.nodes)
    errors = _profile_errors(args, nodes)
    if errors:
        return _report(errors, args.nodes)
    node = topology.node_by_name(nodes, args.node)
    env = rendered_environment(nodes, node, transport=args.transport, profile=args.profile)
    source = args.probe_source or str(Path(__file__).parent / "probe" / "collectives.py")
    command = docker_probe_command(
        image=args.image, environment=env, rank=node["rank"], world_size=len(nodes["nodes"]),
        master_addr=topology.head_node(nodes)["management_ip"], master_port=args.port,
        transport=args.transport, probe_source=source, extra_args=tuple(args.probe_args or ()),
    )
    print("# Plan only. Run on this node during a coordinated window with serving stopped.")
    print(shlex.join(command))
    return 0


def cmd_probe_collectives(args) -> int:
    from sparknet.probe.collectives import main as probe_main

    return probe_main(args.probe_args)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sparknet", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"sparknet {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    profiles = list(nccl_profiles.PROFILES)

    t = sub.add_parser("topology", help="node maps and cabling").add_subparsers(dest="subcommand", required=True)
    v = t.add_parser("validate", help="check a node map for a transport")
    v.add_argument("nodes")
    v.add_argument("--transport", required=True, choices=topology.TRANSPORTS)
    v.add_argument("--mesh-paths", type=int, default=2)
    v.set_defaults(func=cmd_topology_validate)
    r = t.add_parser("render", help="per-node environment for a transport and profile")
    r.add_argument("nodes")
    r.add_argument("node")
    r.add_argument("--transport", required=True, choices=topology.TRANSPORTS)
    r.add_argument("--profile", choices=profiles)
    r.add_argument("--mesh-paths", type=int, default=2)
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_topology_render)
    e = t.add_parser("example", help="print a documentation node map")
    e.add_argument("name")
    e.set_defaults(func=cmd_topology_example)
    d = t.add_parser("discover", help="read cabling over LLDP or rails from sysfs (ssh, read-only) and generate configs")
    d.add_argument("hosts", nargs="+")
    d.add_argument("--ssh-user")
    d.add_argument("--out")
    d.add_argument("--fabric", choices=("cabled", "switched"), default="cabled")
    d.add_argument("--management-ip", action="append", metavar="HOST=IP")
    d.add_argument("--management-interface")
    d.add_argument("--gid-index", type=int, default=3)
    d.add_argument("--traffic-class", type=int)
    d.set_defaults(func=cmd_topology_discover)
    i = t.add_parser("inventory", help="this host's RDMA devices from sysfs")
    i.add_argument("--gid-index", type=int, default=3)
    i.set_defaults(func=cmd_topology_inventory)

    n = sub.add_parser("nccl", help="NCCL profiles and patches").add_subparsers(dest="subcommand", required=True)
    ne = n.add_parser("env", help="print a profile's environment")
    ne.add_argument("--profile", required=True, choices=profiles)
    ne.add_argument("--json", action="store_true")
    ne.set_defaults(func=cmd_nccl_env)
    nv = n.add_parser("validate", help="check a profile, optionally with overrides from an env file")
    nv.add_argument("--profile", required=True, choices=profiles)
    nv.add_argument("--env-file")
    nv.add_argument("--nodes", type=int, help="node count to validate against (default: the profile's first)")
    nv.add_argument("--unpatched", action="store_true", help="the NCCL build lacks the sparknet/nccl/patches series")
    nv.set_defaults(func=cmd_nccl_validate)
    n.add_parser("profiles", help="list profiles").set_defaults(func=cmd_nccl_profiles)
    np_ = n.add_parser("patches", help="print the packaged NCCL patch series, or export it for an image build")
    np_.add_argument("--export", metavar="DIR", help="write the series file and every patch to DIR")
    np_.set_defaults(func=cmd_nccl_patches)

    po = sub.add_parser("policy", help="collective policy").add_subparsers(dest="subcommand", required=True)
    ps = po.add_parser("show", help="the policy of a profile, or of this environment")
    ps.add_argument("--profile", choices=profiles)
    ps.set_defaults(func=cmd_policy_show)

    pr = sub.add_parser("probe", help="fabric checks and the collective probe").add_subparsers(dest="subcommand", required=True)
    pd = pr.add_parser("doctor", help="node map against this host, read-only")
    pd.add_argument("nodes")
    pd.add_argument("--node", required=True)
    pd.add_argument("--transport", required=True, choices=topology.TRANSPORTS)
    pd.set_defaults(func=cmd_probe_doctor)
    pg = pr.add_parser("gpudirect", help="GPU-initiated transport readiness")
    pg.add_argument("--json", action="store_true")
    pg.add_argument("--gpunetio-dir")
    pg.set_defaults(func=cmd_probe_gpudirect)
    rc = pr.add_parser("render-command", help="the bounded container command for one rank's probe")
    rc.add_argument("nodes")
    rc.add_argument("node")
    rc.add_argument("--transport", required=True, choices=topology.TRANSPORTS)
    rc.add_argument("--profile", choices=profiles)
    rc.add_argument("--image", required=True)
    rc.add_argument("--port", type=int, default=29650)
    rc.add_argument("--probe-source")
    rc.add_argument("probe_args", nargs="*", help="arguments after -- go to the probe (e.g. --benchmark)")
    rc.set_defaults(func=cmd_probe_render_command)
    pc = pr.add_parser("collectives", help="run the probe on this rank (needs torch)")
    pc.add_argument("probe_args", nargs=argparse.REMAINDER)
    pc.set_defaults(func=cmd_probe_collectives)
    return p


def split_pass_through(argv: list[str]) -> tuple[list[str], list[str] | None]:
    """Split ``argv`` at the first bare ``--``: the words after it are passed through untouched.

    argparse handles ``--`` differently across the supported Python versions
    (3.10 rejects ``probe render-command ... -- --benchmark``), so the CLI
    takes the separator out before parsing.
    """
    if "--" not in argv:
        return argv, None
    index = argv.index("--")
    return argv[:index], argv[index + 1:]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    head, passed = split_pass_through(argv)
    args = parser().parse_args(head)
    if passed is not None:
        if args.command == "probe" and args.subcommand == "collectives":
            args.probe_args = [*args.probe_args, "--", *passed]
        elif hasattr(args, "probe_args"):
            args.probe_args = [*args.probe_args, *passed]
        else:
            parser().error(f"unexpected arguments after --: {' '.join(passed)}")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
