# Examples

- `minimal_engine.py`: the smallest engine that uses the library the way a
  serving recipe should (setup exchange over gloo, policy-driven dispatch,
  `prepare` before capture, graph replay, `check_health` after each step,
  freeze after warm-up). Run one process per Spark with the environment
  that `sparknet topology render` prints for that node, inside the bounded
  container that `sparknet probe render-command` shows.

The integration guide is `docs/integration.md`.
