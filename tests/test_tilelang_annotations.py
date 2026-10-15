"""TileLang evaluates a prim_func's annotations, and a symbolic shape is a local of the
jitted function: a module with ``from __future__ import annotations`` turns them into
strings that no longer resolve (the TileLang trace clock failed every traced runtime's
setup with ``NameError: name 'n' is not defined``)."""

import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "sparknet"


class TileLangAnnotationsTest(unittest.TestCase):
    def test_no_prim_func_module_postpones_annotations(self) -> None:
        offenders = []
        for path in sorted(PACKAGE.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            postponed = any(line.startswith("from __future__ import") and "annotations" in line
                            for line in source.splitlines())
            if "T.prim_func" in source and postponed:
                offenders.append(str(path.relative_to(PACKAGE.parent)))
        self.assertTrue(any("T.prim_func" in p.read_text(encoding="utf-8") for p in PACKAGE.rglob("*.py")))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
