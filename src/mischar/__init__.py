"""mischar — Legal Citation Mischaracterization Detector.

A five-stage classification pipeline (parse → resolve → attribute → retrieve →
classify) that detects when legal briefs mischaracterize the holdings of cited
cases.
"""

from mischar.constants import DISCLAIMER, PIPELINE_VERSION

__version__ = PIPELINE_VERSION
# Controls what gets exported if someone does from mischar import *. Without this,
# a wildcard import would pull in everything, which can cause name collisions
__all__ = ["DISCLAIMER", "PIPELINE_VERSION", "__version__"]
