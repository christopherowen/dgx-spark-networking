"""CPU Gloo integration check; requires PyTorch, never initializes CUDA.

Run explicitly with: python -m unittest discover -s tests/integration
"""
import os
import socket
import subprocess
import sys
import unittest

CHILD = r"""
import sys
from datetime import timedelta
import torch
import torch.distributed as dist
from sparknet.probe import failstop
class ReachedCollective(Exception):
    pass
def cpu_collective(rank, world):
    value = torch.tensor([rank + 1])
    dist.all_reduce(value)
    assert value.item() == world * (world + 1) // 2
    assert not torch.cuda.is_initialized()
    dist.destroy_process_group()
    raise ReachedCollective()
failstop.eager = cpu_collective
failstop.timedelta = lambda seconds: timedelta(seconds=3)
try:
    failstop.main(['eager', '--rank', sys.argv[1], '--world-size', '2',
                  '--master-addr', '127.0.0.1', '--master-port', sys.argv[2]])
except ReachedCollective:
    print('PASS: isolated CPU Gloo group; CUDA uninitialized', flush=True)
"""

class FailstopRendezvousTest(unittest.TestCase):
    def test_separate_group_ignores_inherited_agent_store(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        env = dict(os.environ, TORCHELASTIC_USE_AGENT_STORE='True',
                   TORCHELASTIC_RESTART_COUNT='0')
        children = [subprocess.Popen([sys.executable, '-c', CHILD, str(rank), str(port)],
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    for rank in range(2)]
        results = []
        try:
            for child in children:
                output, _ = child.communicate(timeout=20)
                results.append((child.returncode, output))
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.communicate(timeout=5)
        self.assertEqual([code for code, _ in results], [0, 0], results)
        for _, output in results:
            self.assertIn('PASS: isolated CPU Gloo group; CUDA uninitialized', output)

if __name__ == '__main__':
    unittest.main()
