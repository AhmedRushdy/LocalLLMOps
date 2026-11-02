import os

# Keep the background health prober out of unit tests; they mock the
# request client and must not depend on live backends.
os.environ.setdefault("HEALTH_PROBE_INTERVAL", "0")
