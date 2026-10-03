"""Install only the explicit private incumbent-rebuild guard before vLLM."""
import os
import sys

if os.environ.get("MEGARTX_M1_PRIVATE_AOT"):
    try:
        from m1_private_aot import install_guard
        install_guard()
    except BaseException as error:
        # CPython normally prints sitecustomize exceptions and continues.
        # A cache admission error must stop this process before model dispatch.
        print("Private AOT startup rejected: " + str(error), file=sys.stderr, flush=True)
        os._exit(78)
