#!/usr/bin/env python3
"""Complete Astra Demo - Everything We Built.
========================================

Run all components together to showcase the full platform.
"""

import contextlib
import sys

sys.path.append("/mnt/f/astra/astra-main")


with contextlib.suppress(Exception):
    exec(open("test_foundation.py").read())

with contextlib.suppress(Exception):
    exec(open("examples/monte_carlo_demo.py").read())

with contextlib.suppress(Exception):
    exec(open("examples/risk_management_demo.py").read())

