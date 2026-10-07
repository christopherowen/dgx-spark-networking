"""CPU protocol stress of the production C proxy and the launcher geometry; does not qualify GPU/RDMA.

The simulator in ``tests/oneshot_sim`` includes the real ``_roce_proxy.c``
and replaces only libibverbs and the GPU endpoint. The Python checks execute
the production topology resolver and the kernels' flag-selection expressions
in isolation, because importing the runtime needs torch and the CuTe DSL.
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

    def test_proxy_abi_is_the_qualified_one(self):
        source = (ROCE / "_roce_proxy.c").read_text()
        self.assertIn("#define ROCE_ABI_VERSION 10", source)
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
    def test_family_selection(self):
        kernels = _module(ROCE / "_kernels.py")
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(kernels.family(), "cute")
            self.assertEqual(kernels.family("tilelang"), "tilelang")
        with unittest.mock.patch.dict(os.environ, {"SPARKNET_ROCE_KERNELS": "tilelang"}):
            self.assertEqual(kernels.family(), "tilelang")
            self.assertEqual(kernels.family("cute"), "cute")
        with self.assertRaises(ValueError):
            kernels.family("triton")
        self.assertEqual(kernels.FAMILIES, ("cute", "tilelang"))

    def test_device_header_defines_every_function_the_kernels_call(self):
        device = _module(ROCE / "_device.py")
        source = device.render_source(world_size=4, rank=1, slots=2, flag_stride=128, hca_count=4, neighbor_lanes=2)
        for name in device.FUNCTIONS:
            self.assertRegex(source, rf"[ *]{name}\(", name)
        for define in ("#define ROCE_WORLD 4", "#define ROCE_RANK 1", "#define ROCE_SLOTS 2",
                       "#define ROCE_FLAG_STRIDE 128", "#define ROCE_HCA_COUNT 4", "#define ROCE_NEIGHBOR_LANES 2"):
            self.assertIn(define, source)
        self.assertEqual(source.count("{"), source.count("}"), "unbalanced braces in the device header")
        # The same PTX as the CuTe intrinsics, so both families order memory and round the same way.
        intrinsics = (ROCE / "_cute_intrinsics.py").read_text()
        for ptx in ("ld.relaxed.gpu.global.u32", "ld.relaxed.sys.global.u32", "atom.relaxed.gpu.global.add.u32",
                    "st.release.gpu.global.u32", "st.relaxed.sys.global.u32", "fence.sc.sys;", "fence.sc.gpu;",
                    "ld.acquire.sys.global.u32", "ld.relaxed.sys.global.v4.u32", "ld.global.v4.u32", "st.global.v4.u32",
                    "cvt.f32.bf16", "cvt.f32.f16", "cvt.rn.bf16.f32", "cvt.rn.f16x2.f32"):
            self.assertIn(ptx, source, ptx)
            self.assertIn(ptx, intrinsics, ptx)
        # Every extern the TileLang kernels name exists in the header.
        known = device.FUNCTIONS + device.FUSED_FUNCTIONS
        for module_name in ("_oneshot_tilelang.py", "_allgather_tilelang.py", "_fused_tilelang.py"):
            text = (ROCE / module_name).read_text()
            for name in sorted(set(re.findall(r'T\.call_extern\("(roce_[a-z0-9_]+)"', text))):
                self.assertIn(name, known, (module_name, name))
            for name in sorted(set(re.findall(r'"(roce_(?:fused_)?reduce_pack_[a-z0-9]+)"', text))):
                self.assertIn(name, known, (module_name, name))

    def test_fused_header_carries_the_runtime_layout(self):
        device = _module(ROCE / "_device.py")
        plain = device.render_source(world_size=4, rank=1, slots=2, flag_stride=128, hca_count=4, neighbor_lanes=2)
        self.assertIn("#define ROCE_TRACE 0", plain)
        self.assertNotIn("#define ROCE_FUSED", plain)
        constants = {name: 1000 + index for index, name in enumerate(device.FUSED_CONSTANTS)}
        fused = device.render_source(world_size=4, rank=1, slots=2, flag_stride=128, hca_count=4, neighbor_lanes=2,
                                     fused=constants)
        self.assertIn("#define ROCE_FUSED 1", fused)
        for name, value in constants.items():
            self.assertIn(f"#define {name} {value}", fused)
        for name in device.FUSED_FUNCTIONS:
            self.assertRegex(fused, rf"[ *]{name}\(", name)
        # The fused functions sit inside the #ifdef ROCE_FUSED section.
        section = fused[fused.index("#ifdef ROCE_FUSED"):]
        section = section[:section.index("#endif")]
        for name in device.FUSED_FUNCTIONS:
            self.assertRegex(section, rf"[ *]{name}\(", name)
        with self.assertRaises(ValueError):
            device.render_source(world_size=4, rank=1, slots=2, flag_stride=128, hca_count=4, neighbor_lanes=2,
                                 fused={"ROCE_SEND_OFF": 0})
        # The fused counters reset themselves, so any grid size works.
        arrive = section[section.index("roce_fused_arrive_last("):]
        self.assertIn("prior + 1u == grid", arrive[:400])
        self.assertIn("roce_st_relaxed_gpu_u32(counter, 0u)", arrive[:400])

    def test_runtime_reserves_the_fused_counters_after_the_poison_word(self):
        runtime = (ROCE / "runtime.py").read_text()
        self.assertIn("4 + 2 * self._counter_classes", runtime)
        self.assertIn("return 1 + 2 * self._counter_classes", runtime)  # poison
        self.assertIn("return 2 + 2 * self._counter_classes", runtime)  # fused stage
        self.assertIn("return 3 + 2 * self._counter_classes", runtime)  # fused tail
        for name in ("ROCE_POISON_INDEX", "ROCE_FUSED_STAGE_INDEX", "ROCE_FUSED_TAIL_INDEX", "ROCE_SPIN_LIMIT"):
            self.assertIn(f'"{name}"', runtime)
        # Every other collective refuses to run between the halves.
        for call in ('_refuse_while_pending("all_reduce")', '_refuse_while_pending("all_gather")',
                     '_refuse_while_pending("send")', '_refuse_while_pending("fused_send")'):
            self.assertIn(call, runtime)

    def test_tilelang_modules_import_without_a_gpu_stack(self):
        for module_name in ("_oneshot_tilelang.py", "_allgather_tilelang.py", "_fused_tilelang.py", "_kernels.py",
                            "_device.py", "_freeze.py"):
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
        for module_name in ("_oneshot_tilelang.py", "_allgather_tilelang.py", "_fused_tilelang.py"):
            text = (ROCE / module_name).read_text()
            self.assertEqual(text.count("T.annotate_min_blocks_per_sm(resident)"), text.count("T.Kernel("), module_name)
            self.assertIn("_resident_blocks(threads)", text, module_name)

    def test_runtime_routes_launchers_through_the_family(self):
        runtime = (ROCE / "runtime.py").read_text()
        self.assertIn("reduce_launcher(self.kernel_family,", runtime)
        self.assertIn("gather_launcher(self.kernel_family,", runtime)
        self.assertIn('"kernel_family": self.kernel_family', runtime)
        self.assertNotIn("_oneshot_cute import", runtime)


class TopologyResolverTest(unittest.TestCase):
    def setUp(self):
        namespace = dict(vars(typing), os=os, json=json, MAX_STRIPES=2, MAX_LOCAL_HCAS=4,
                         ENV_PEER_HCAS=("SPARKNET_ROCE_PEER_HCAS",),
                         discover_hcas=lambda gid_index=None: ())
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
                                                      peer_hca_names=routes, gid_index=3, topology="ring4")
                self.assertEqual((len(names), stripes), (2 * width, width))
                self.assertEqual(result[rank], ())
                self.assertEqual(result[(rank + 2) % 4], ())
                self.assertEqual({p: result[p] for p in peers}, routes)

    def test_invalid_geometry_rejected(self):
        for world, routes in ((3, {1: ("a",), 2: ("b",)}), (4, None), (4, {1: ("a",), 2: ("b",), 3: ("c",)})):
            with self.assertRaises(ValueError):
                self.resolve(world_size=world, rank=0, hca_names=("a", "b"), peer_hca_names=routes, gid_index=3, topology="ring4")
        with self.assertRaises(ValueError):
            self.resolve(world_size=3, rank=0, hca_names=None, peer_hca_names={1: ("a",), 2: ("b", "c")}, gid_index=3)
        with self.assertRaises(ValueError):
            self.resolve(world_size=3, rank=0, hca_names=("a", "b"), peer_hca_names={1: ("a",), 2: ("c",)}, gid_index=3)

    def test_direct_clique_and_mesh_routes(self):
        names, routes, stripes = self.resolve(world_size=3, rank=1, hca_names=("mlx5_0", "mlx5_1"), peer_hca_names=None, gid_index=3)
        self.assertEqual((names, stripes), (("mlx5_0", "mlx5_1"), 2))
        self.assertEqual(routes, (("mlx5_0", "mlx5_1"), (), ("mlx5_0", "mlx5_1")))
        for rank in range(4):
            full = {p: (f"h{p % 2}a", f"h{p % 2}b") for p in range(4) if p != rank}
            names, result, stripes = self.resolve(world_size=4, rank=rank, hca_names=None, peer_hca_names=full, gid_index=3, topology="mesh4")
            self.assertEqual(stripes, 2)
            self.assertTrue(all(result[p] == full[p] for p in full))
            four = {(rank + 1) % 4: ("a", "b"), (rank - 1) % 4: ("c", "d"), (rank + 2) % 4: ("a", "c", "b", "d")}
            names, result, slots = self.resolve(world_size=4, rank=rank, hca_names=None, peer_hca_names=four, gid_index=3, topology="mesh4")
            self.assertEqual((slots, len(names), len(result[(rank + 2) % 4])), (4, 4, 4))

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

    def test_launchers_wait_on_the_relay_flags(self):
        for module, class_name in (("_oneshot_cute.py", "_RoceOneshotLaunch"), ("_allgather_cute.py", "_RoceAllGatherLaunch")):
            source = ROCE / module
            tree = ast.parse(source.read_text())
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
            kernel = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "kernel")
            cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
            key = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_process_key")
            namespace = {"_DTYPE_PACK_ELEMS": {"bfloat16": 8}}
            exec(compile(ast.Module(body=[cls, key], type_ignores=[]), str(source), "exec"), namespace)
            prefix = ("bfloat16",) if module == "_oneshot_cute.py" else ()
            wait = next(n for n in ast.walk(kernel) if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                        and "self._world_size * self._hca_count" in ast.unparse(n.test))
            select = ast.Module(body=wait.body[:4], type_ignores=[])
            for width in (1, 2):
                for rank in range(4):
                    args = prefix + (4, rank, 128, 2, 64, width)
                    launch = namespace[class_name](*args, True)
                    self.assertEqual((launch._hca_count, launch._neighbor_lanes), (2 * width, width))
                    self.assertNotEqual(namespace["_process_key"](*args, 0, True), namespace["_process_key"](*args, 0, False))
                    selected = []
                    for tid in range(4 * 2 * width):
                        env = dict(self=launch, tidx=tid, Int32=int, cutlass=types.SimpleNamespace(const_expr=lambda x: x))
                        exec(compile(select, str(source), "exec"), env)
                        if env["active"]:
                            selected.append((env["peer"], env["hca"]))
                    expected = [(p, lane) for p in range(4) if p != rank for lane in range(2 * width if p == (rank + 2) % 4 else width)]
                    self.assertEqual(selected, expected)
            for width in (1, 2, 4):
                launch = namespace[class_name](*(prefix + (4, 0, 128, 2, 64, width)))
                self.assertEqual(launch._hca_count, width)
                self.assertEqual(launch._neighbor_lanes, 2 if width == 4 else width)

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


if __name__ == "__main__":
    unittest.main()
