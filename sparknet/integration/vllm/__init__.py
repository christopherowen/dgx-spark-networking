"""vLLM device-communicator adapter for RoCEnante (see README.md in this directory)."""

from .roce_all_reduce import REQUIRED_API_VERSION, SparknetRoceAllReduce

__all__ = ["REQUIRED_API_VERSION", "SparknetRoceAllReduce"]
