"""CPU protocol stress of the production C proxy and the launcher geometry; does not qualify GPU/RDMA.

The simulator in ``tests/oneshot_sim`` includes the real ``_roce_proxy.c``
and replaces only libibverbs and the GPU endpoint. The Python checks execute
the production topology resolver in isolation, and compile the generated
header's flag-selection code with the host compiler, because importing the
runtime needs torch and TileLang.
"""

import ast
import json
import os
import re
import shutil
import subprocess
import tempfile
import types
import typing
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "tests" / "oneshot_sim"
ROCE = ROOT / "sparknet" / "oneshot"


def _functions(path: Path, names: set[str], namespace: dict) -> dict:
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class ProxySimulatorTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which(os.environ.get("CC", "cc")), "needs a C compiler")
    def test_simulated_collectives_pass_under_sanitizers(self):
        with tempfile.TemporaryDirectory(prefix="roce-proxy-") as tmp:
            binary = str(Path(tmp) / "simulate")
            build = subprocess.run([os.environ.get("CC", "cc"), "-O1", "-g", "-std=gnu11", "-Wall", "-Wextra", "-Werror",
                                    "-fsanitize=address,undefined", "-pthread", "-I" + str(SIM), str(SIM / "simulate.c"),
                                    "-o", binary], capture_output=True, text=True, timeout=120)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([binary], capture_output=True, text=True, timeout=300)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("PASS: direct3/ring4/mesh4", run.stdout)
            self.assertIn("thread placement", run.stdout)
            self.assertIn("GID selection by address", run.stdout)

    def test_proxy_abi_is_the_qualified_one(self):
        source = (ROCE / "_roce_proxy.c").read_text()
        self.assertIn("#define ROCE_ABI_VERSION 10", source)
        # Each HCA's source GID comes from the table scan; no slot is taken as given.
        self.assertIn("ibv_query_gid_ex(hca->ctx", source)
        self.assertNotIn("ibv_query_gid(", source)
        self.assertIn("rtr.ah_attr.grh.sgid_index = (uint8_t)hca->gid_index;", source)
        self.assertIn("lib.roce_abi_version() != 10", (ROCE / "_proxy.py").read_text())
        self.assertIn('getenv("SPARKNET_ROCE_MESH_ROTATE")', source)
        self.assertIn('getenv("SPARKNET_ROCE_PROXY_CPU")', source)
        self.assertNotIn("b12x.", (ROCE / "runtime.py").read_text())


def _module(path: Path):
    """Import one module file directly (the package __init__ needs torch)."""
    import importlib.util

    import sys

    spec = importlib.util.spec_from_file_location(f"sparknet_cpu_test.{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve postponed annotations through sys.modules
    spec.loader.exec_module(module)
    return module


class KernelFamilyTest(unittest.TestCase):
    def test_the_family_switch_is_refused(self):
        """One kernel family since 0.4.0: the retired switch is an error, not a no-op."""
        kernels = _module(ROCE / "_kernels.py")
        self.assertEqual(kernels.kernel_setting_problems({}), [])
        for value in ("cute", "tilelang"):
            [message] = kernels.kernel_setting_problems({"SPARKNET_ROCE_KERNELS": value})
            self.assertIn("unset it", message)
        for retired in ("_oneshot_cute.py", "_allgather_cute.py", "_cute_intrinsics.py", "_compile.py"):
            self.assertFalse((ROCE / retired).exists(), retired)

    def test_device_header_defines_every_function_the_kernels_call(self):
        device = _module(ROCE / "_device.py")
        source = device.render_source(world_size=4, rank=1, slots=2, flag_stride=128, hca_count=4, neighbor_lanes=2)
        for name in device.FUNCTIONS:
            self.assertRegex(source, rf"[ *]{name}\(", name)
        for define in ("#define ROCE_WORLD 4", "#define ROCE_RANK 1", "#define ROCE_SLOTS 2",
                       "#define ROCE_FLAG_STRIDE 128", "#define ROCE_HCA_COUNT 4", "#define ROCE_NEIGHBOR_LANES 2"):
            self.assertIn(define, source)
        self.assertEqual(source.count("{"), source.count("}"), "unbalanced braces in the device header")
        # The protocol's memory ordering and rounding, as the qualified kernels spell them.
        for ptx in ("ld.relaxed.gpu.global.u32", "ld.relaxed.sys.global.u32", "atom.relaxed.gpu.global.add.u32",
                    "st.release.gpu.global.u32", "st.relaxed.sys.global.u32", "fence.sc.sys;", "fence.sc.gpu;",
                    "ld.acquire.sys.global.u32", "ld.relaxed.sys.global.v4.u32", "ld.global.v4.u32", "st.global.v4.u32",
                    "cvt.f32.bf16", "cvt.f32.f16", "cvt.rn.bf16.f32", "cvt.rn.f16x2.f32"):
            self.assertIn(ptx, source, ptx)
        # Every extern the TileLang kernels name exists in the header.
        for module_name in ("_oneshot_tilelang.py", "_allgather_tilelang.py"):
            text = (ROCE / module_name).read_text()
            for name in sorted(set(re.findall(r'T\.call_extern\("(roce_[a-z0-9_]+)"', text))):
                self.assertIn(name, device.FUNCTIONS, (module_name, name))
            for name in sorted(set(re.findall(r'"(roce_reduce_pack_[a-z0-9]+)"', text))):
                self.assertIn(name, device.FUNCTIONS, (module_name, name))

    def test_tilelang_modules_import_without_a_gpu_stack(self):
        for module_name in ("_oneshot_tilelang.py", "_allgather_tilelang.py", "_kernels.py", "_device.py", "_freeze.py"):
            tree = ast.parse((ROCE / module_name).read_text())
            top_level = {alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
            top_level |= {(node.module or "").split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom) and node.level == 0}
            self.assertFalse(top_level & {"torch", "tilelang", "cutlass", "cuda", "tvm"}, (module_name, top_level))
        launch = _module(ROCE / "_freeze.py")
        launch.freeze_kernel_resolution("test")
        with self.assertRaises(launch.KernelResolutionFrozenError):
            launch.raise_if_kernel_resolution_frozen("tilelang.jit")
        launch.thaw_kernel_resolution()
        launch.raise_if_kernel_resolution_frozen("tilelang.jit")

    def test_tilelang_kernels_declare_full_residency(self):
        # Without a minimum blocks-per-SM bound nvcc spends 54 to 56 registers and the
        # all-reduce loses ~15 us per launch in a decode step (2026-10-05 profiles).
        for module_name in ("_oneshot_tilelang.py", "_allgather_tilelang.py"):
            text = (ROCE / module_name).read_text()
            self.assertIn("T.annotate_min_blocks_per_sm(resident)", text, module_name)
            self.assertIn("_resident_blocks(threads)", text, module_name)

    def test_runtime_routes_launchers_to_the_tilelang_kernels(self):
        runtime = (ROCE / "runtime.py").read_text()
        self.assertIn("reduce_launcher(*self._launcher_key(dtype))", runtime)
        self.assertIn("gather_launcher(*self._gather_launcher_key())", runtime)
        self.assertIn("kernel_setting_problems(os.environ)", runtime)
        self.assertIn('if "kernels" in _ignored:', runtime)
        self.assertNotIn("kernel_family", runtime)
        kernels = (ROCE / "_kernels.py").read_text()
        self.assertIn("from ._oneshot_tilelang import get_launcher", kernels)
        self.assertIn("from ._allgather_tilelang import get_launcher", kernels)


class TopologyResolverTest(unittest.TestCase):
    def setUp(self):
        namespace = dict(vars(typing), os=os, json=json, MAX_STRIPES=2, MAX_LOCAL_HCAS=4,
                         ENV_PEER_HCAS=("SPARKNET_ROCE_PEER_HCAS",),
                         discover_hcas=lambda: ())
        _functions(ROCE / "runtime.py", {"_resolve_hca_topology", "_peer_hca_names_from_env"}, namespace)
        self.resolve = namespace["_resolve_hca_topology"]
        self.from_env = namespace["_peer_hca_names_from_env"]
        os.environ.pop("SPARKNET_ROCE_PEER_HCAS", None)

    def test_ring4_routes_only_neighbours(self):
        for rank in range(4):
            for width in (1, 2):
                peers = ((rank - 1) % 4, (rank + 1) % 4)
                routes = {p: tuple(f"hca{p}_{i}" for i in range(width)) for p in peers}
                names, result, stripes = self.resolve(world_size=4, rank=rank, hca_names=None,
                                                      peer_hca_names=routes, topology="ring4")
                self.assertEqual((len(names), stripes), (2 * width, width))
                self.assertEqual(result[rank], ())
                self.assertEqual(result[(rank + 2) % 4], ())
                self.assertEqual({p: result[p] for p in peers}, routes)

    def test_invalid_geometry_rejected(self):
        for world, routes in ((3, {1: ("a",), 2: ("b",)}), (4, None), (4, {1: ("a",), 2: ("b",), 3: ("c",)})):
            with self.assertRaises(ValueError):
                self.resolve(world_size=world, rank=0, hca_names=("a", "b"), peer_hca_names=routes, topology="ring4")
        with self.assertRaises(ValueError):
            self.resolve(world_size=3, rank=0, hca_names=None, peer_hca_names={1: ("a",), 2: ("b", "c")})
        with self.assertRaises(ValueError):
            self.resolve(world_size=3, rank=0, hca_names=("a", "b"), peer_hca_names={1: ("a",), 2: ("c",)})

    def test_direct_clique_and_mesh_routes(self):
        names, routes, stripes = self.resolve(world_size=3, rank=1, hca_names=("mlx5_0", "mlx5_1"), peer_hca_names=None)
        self.assertEqual((names, stripes), (("mlx5_0", "mlx5_1"), 2))
        self.assertEqual(routes, (("mlx5_0", "mlx5_1"), (), ("mlx5_0", "mlx5_1")))
        for rank in range(4):
            full = {p: (f"h{p % 2}a", f"h{p % 2}b") for p in range(4) if p != rank}
            names, result, stripes = self.resolve(world_size=4, rank=rank, hca_names=None, peer_hca_names=full, topology="mesh4")
            self.assertEqual(stripes, 2)
            self.assertTrue(all(result[p] == full[p] for p in full))
            four = {(rank + 1) % 4: ("a", "b"), (rank - 1) % 4: ("c", "d"), (rank + 2) % 4: ("a", "c", "b", "d")}
            names, result, slots = self.resolve(world_size=4, rank=rank, hca_names=None, peer_hca_names=four, topology="mesh4")
            self.assertEqual((slots, len(names), len(result[(rank + 2) % 4])), (4, 4, 4))

    def test_subnet_map_gives_every_opened_hca_its_network(self):
        import ipaddress

        ns = _functions(ROCE / "runtime.py", {"_subnets_from_env", "_gid_networks"},
                        dict(vars(typing), os=os, json=json, ipaddress=ipaddress, ANY_IPV4="0.0.0.0/0",
                             ENV_SUBNETS=("SPARKNET_ROCE_SUBNETS",)))
        parse, networks = ns["_subnets_from_env"], ns["_gid_networks"]
        os.environ.pop("SPARKNET_ROCE_SUBNETS", None)
        self.assertIsNone(parse())
        # No map: any IPv4 address qualifies, and the proxy still needs exactly one.
        self.assertEqual(networks(("a", "b"), None), (ipaddress.IPv4Network("0.0.0.0/0"),) * 2)
        os.environ["SPARKNET_ROCE_SUBNETS"] = '{"rocep1s0f0": "10.12.1.0/24", "roceP2p1s0f0": "10.12.2.0/24"}'
        subnets = parse()
        self.assertEqual(networks(("roceP2p1s0f0", "rocep1s0f0"), subnets),
                         (ipaddress.IPv4Network("10.12.2.0/24"), ipaddress.IPv4Network("10.12.1.0/24")))
        with self.assertRaisesRegex(ValueError, "SPARKNET_ROCE_SUBNETS names no subnet for rocep1s0f1"):
            networks(("rocep1s0f0", "rocep1s0f1"), subnets)
        for bad in ('["10.12.1.0/24"]', "not json", '{"a": "10.12.1.4/24"}', '{"a": 5}', '{"a": "fe80::/64"}'):
            os.environ["SPARKNET_ROCE_SUBNETS"] = bad
            with self.assertRaisesRegex(ValueError, "SPARKNET_ROCE_SUBNETS"):
                parse()
        os.environ.pop("SPARKNET_ROCE_SUBNETS", None)

    def test_peer_map_environment(self):
        self.assertIsNone(self.from_env())
        os.environ["SPARKNET_ROCE_PEER_HCAS"] = '{"0": ["mlx5_0", "mlx5_1"], "2": ["mlx5_2", "mlx5_3"]}'
        self.assertEqual(self.from_env(), {0: ("mlx5_0", "mlx5_1"), 2: ("mlx5_2", "mlx5_3")})
        os.environ["SPARKNET_ROCE_PEER_HCAS"] = "not json"
        with self.assertRaisesRegex(ValueError, "SPARKNET_ROCE_PEER_HCAS"):
            self.from_env()
        os.environ.pop("SPARKNET_ROCE_PEER_HCAS", None)


class LauncherGeometryTest(unittest.TestCase):
    """Execute the kernels' actual flag-selection expressions against the simulator's geometry."""

    @unittest.skipUnless(shutil.which(os.environ.get("CC", "cc")), "needs a C compiler")
    def test_launchers_wait_on_the_relay_flags(self):
        """Compile the generated header's own flag selection for every rank and lane geometry
        and check which (peer, lane) flags a launch waits on."""
        device = _module(ROCE / "_device.py")
        lanes = _functions(ROCE / "_oneshot_tilelang.py", {"_lanes"}, {})["_lanes"]
        for ring4, widths in ((True, (1, 2)), (False, (1, 2, 4))):
            for width in widths:
                hca_count, neighbor_lanes = lanes(width, ring4)
                if ring4:
                    self.assertEqual((hca_count, neighbor_lanes), (2 * width, width))
                else:
                    self.assertEqual((hca_count, neighbor_lanes), (width, 2 if width == 4 else width))
                for rank in range(4):
                    header = device.render_source(world_size=4, rank=rank, slots=2, flag_stride=128,
                                                  hca_count=hca_count, neighbor_lanes=neighbor_lanes)
                    wait = header[header.index("__device__ __forceinline__ void roce_wait_flags("):]
                    select = wait[wait.index("int peer = tidx / ROCE_HCA_COUNT;"):wait.index("if (active) {")]
                    defines = "".join(line + "\n" for line in header.splitlines()
                                      if line.startswith("#define ROCE_") and "(" not in line.split()[1])
                    program = (f"#include <stdbool.h>\n#include <stdio.h>\n{defines}"
                               "int main(void) {\n"
                               "    for (int tidx = 0; tidx < ROCE_WORLD * ROCE_HCA_COUNT; tidx++) {\n"
                               f"        {select}\n"
                               "        if (active) printf(\"%d %d\\n\", peer, hca);\n"
                               "    }\n    return 0;\n}\n")
                    with tempfile.TemporaryDirectory() as tmp:
                        (Path(tmp) / "select.c").write_text(program)
                        subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Werror", "-o",
                                        str(Path(tmp) / "select"), str(Path(tmp) / "select.c")], check=True)
                        out = subprocess.run([str(Path(tmp) / "select")], check=True, capture_output=True, text=True).stdout
                    selected = [tuple(map(int, line.split())) for line in out.splitlines()]
                    # The opposite rank on every lane, the neighbours on their own lanes.
                    expected = [(p, lane) for p in range(4) if p != rank
                                for lane in range(hca_count if p == (rank + 2) % 4 else neighbor_lanes)]
                    self.assertEqual(selected, expected, (ring4, width, rank))

    def test_runtime_launcher_keys_carry_the_relay_mode(self):
        source = (ROCE / "runtime.py").read_text()
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RoceOneshotAllReduce")
        for name in ("_launcher_key", "_gather_launcher_key"):
            fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertIn("self.topology == 'ring4'", ast.unparse(fn))
        prepare = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "prepare")
        self.assertIn("is_current_stream_capturing", ast.unparse(prepare))
        self.assertFalse(any(isinstance(n, ast.FunctionDef) and n.name.startswith("_run_prepared") for n in cls.body))


class FailStopTest(unittest.TestCase):
    """A timed-out wait records why, then stops the GPU; the host reports the record."""

    def _runtime_constants(self) -> dict:
        tree = ast.parse((ROCE / "runtime.py").read_text())
        names = {"DEFAULT_TIMEOUT_MS", "MAX_TIMEOUT_MS", "CTRL_TIMEOUT_WORD", "CTRL_WAITED_WORD", "REMOVED_SETTINGS"}
        body = [n for n in tree.body if isinstance(n, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names for target in n.targets)]
        namespace: dict = {}
        exec(compile(ast.Module(body=body, type_ignores=[]), "runtime.py", "exec"), namespace)
        return namespace

    def test_one_control_record_layout_everywhere(self):
        """The timeout and waited words sit at the same place for the runtime, the
        proxy and the device header."""
        const = self._runtime_constants()
        proxy = (ROCE / "_roce_proxy.c").read_text()
        self.assertIn(f"#define ROCE_CTRL_TIMEOUT_WORD {const['CTRL_TIMEOUT_WORD']}\n", proxy)
        device = (ROCE / "_device.py").read_text()
        self.assertIn(f"#define ROCE_CTRL_TIMEOUT_US {4 * const['CTRL_TIMEOUT_WORD']}\n", device)
        self.assertIn(f"#define ROCE_CTRL_WAITED_US {4 * const['CTRL_WAITED_WORD']}\n", device)
        # Words 0-6 are the doorbell's and the timeout record's; the region holds 32.
        self.assertGreater(const["CTRL_TIMEOUT_WORD"], 6)
        self.assertLess(4 * (const["CTRL_WAITED_WORD"] + 1), 128 + 1)

    def test_timeout_is_milliseconds_within_the_record(self):
        const = self._runtime_constants()
        ns = _functions(ROCE / "runtime.py", {"_checked_timeout_ms"}, dict(const))
        check = ns["_checked_timeout_ms"]
        self.assertEqual(check(10_000), 10_000)
        self.assertEqual(check(const["MAX_TIMEOUT_MS"]), const["MAX_TIMEOUT_MS"])
        self.assertLessEqual(1000 * const["MAX_TIMEOUT_MS"], 2**31 - 1)  # microseconds in an int32 word
        for bad in (0, -1, const["MAX_TIMEOUT_MS"] + 1):
            with self.assertRaises(ValueError):
                check(bad)
        self.assertEqual(check(const["DEFAULT_TIMEOUT_MS"]), 10_000)

    def test_the_poll_count_setting_is_refused_not_ignored(self):
        const = self._runtime_constants()
        ns = _functions(ROCE / "runtime.py", {"_removed_setting_problems"}, dict(const, Mapping=typing.Mapping))
        problems = ns["_removed_setting_problems"]
        self.assertEqual(problems({}), [])
        self.assertEqual(problems({"SPARKNET_ROCE_TIMEOUT_MS": "10000"}), [])
        [message] = problems({"SPARKNET_ROCE_SPIN_LIMIT": "5000000"})
        self.assertIn("SPARKNET_ROCE_SPIN_LIMIT", message)
        self.assertIn("SPARKNET_ROCE_TIMEOUT_MS", message)
        # A fixed GID slot is refused too; NCCL's own NCCL_IB_GID_INDEX is not the runtime's to read.
        [message] = problems({"SPARKNET_ROCE_GID_INDEX": "3", "NCCL_IB_GID_INDEX": "3"})
        self.assertIn("SPARKNET_ROCE_GID_INDEX is no longer read", message)
        self.assertIn("SPARKNET_ROCE_SUBNETS", message)
        runtime = (ROCE / "runtime.py").read_text()
        self.assertNotIn("NCCL_IB_GID_INDEX", runtime)
        self.assertNotIn("gid_index:", runtime)
        self.assertNotIn("self.gid_index ", runtime)

    def _failure_method(self):
        tree = ast.parse((ROCE / "runtime.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RoceOneshotAllReduce")
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"failure", "check_health"}]
        namespace = dict(self._runtime_constants(), Optional=typing.Optional)
        exec(compile(ast.Module(body=methods, type_ignores=[]), "runtime.py", "exec"), namespace)
        return namespace["failure"], namespace["check_health"]

    def _fake_runtime(self, ctrl, proxy_error=None):
        proxy = types.SimpleNamespace(failed=lambda: proxy_error is not None, error=lambda: proxy_error)
        return types.SimpleNamespace(rank=0, timeout_ms=10_000, _ctrl_np=ctrl, _proxy=proxy)

    def test_failure_report_puts_the_kernel_record_before_the_relay(self):
        failure, check_health = self._failure_method()
        const = self._runtime_constants()
        healthy = self._fake_runtime([0] * 9)
        self.assertIsNone(failure(healthy))
        check_health(types.SimpleNamespace(failure=lambda: failure(healthy)))
        ctrl = [0] * 9
        ctrl[2], ctrl[3], ctrl[6], ctrl[const["CTRL_WAITED_WORD"]] = 4855, 2, 1, 10_000_512
        stalled = self._fake_runtime(ctrl, proxy_error="relay stopped at seq 4855: this rank's collective at seq 4855 timed out")
        report = failure(stalled)
        self.assertRegex(report, r"^RoCE collective on rank 0 saw nothing from rank 2, HCA 1, for 10\.001 s "
                                 r"\(timeout 10 s\) at sequence 4855; the GPU was stopped there; RoCE proxy failed: relay stopped")
        with self.assertRaisesRegex(RuntimeError, "saw nothing from rank 2"):
            check_health(types.SimpleNamespace(failure=lambda: failure(stalled)))
        relay_only = self._fake_runtime([0] * 9, proxy_error="relay timed out after 10.000 s: seq 7 pending 0x1")
        self.assertEqual(failure(relay_only), "RoCE proxy failed: relay timed out after 10.000 s: seq 7 pending 0x1")

    def test_the_wait_records_then_fences_then_traps(self):
        """The record (peer, HCA, time waited) before the failed sequence the proxy and
        host watch, each fenced to system scope, and nothing computes after it."""
        device = (ROCE / "_device.py").read_text()
        branch = device[device.index("if (waited != 0u) {"):]
        order = ["ctrl + 12", "ctrl + 24", "ROCE_CTRL_WAITED_US", "roce_fence_sc_sys()", "ctrl + 8, seq",
                 "roce_fence_sc_sys()", "roce_trap()"]
        positions = [branch.index(step, branch.index(order[i - 1]) + 1 if i else 0) for i, step in enumerate(order)]
        self.assertEqual(positions, sorted(positions))
        for module in ("_oneshot_tilelang.py", "_allgather_tilelang.py"):
            source = (ROCE / module).read_text()
            self.assertNotIn("poison", source, module)
            self.assertNotIn("spin_limit", source, module)


if __name__ == "__main__":
    unittest.main()
