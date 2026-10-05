"""Private bounded worker entry; invoke only through the owned analysis launcher."""
import os
from pathlib import Path
import resource
import sys


def main():
    if sys.platform != 'linux' or len(sys.argv) != 3:
        raise ValueError('Owned Linux analysis invocation required')
    seconds = int(sys.argv[1])
    deadline = float(sys.argv[2])
    if not 0 < seconds <= 300:
        raise ValueError('CPU allowance may only be reduced')
    # Linux ignores RLIMIT_RSS. A hard AS bound is stronger, enforceable, and
    # installed before importing the validator or loading any operand arrays.
    resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from megartx.prefill_attention_analysis import worker
    worker(seconds, deadline)


if __name__ == '__main__':
    main()
