"""vLLM device-communicator adapter for one-shot (see README.md in this directory)."""

from .roce_all_reduce import REQUIRED_API_VERSION, SparknetOneShotAllReduce
from .roce_p2p import SparknetOneShotP2P

__all__ = ["REQUIRED_API_VERSION", "SparknetOneShotAllReduce", "SparknetOneShotP2P"]
