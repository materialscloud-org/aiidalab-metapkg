"""Helpful utilities for the AiiDA lab tools."""

import json
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.parse import urlsplit, urlunsplit, parse_qs, urlencode
from collections import defaultdict
from functools import wraps
from threading import Lock

import requests
from IPython.lib import backgroundjobs as bg

from .config import AIIDALAB_APPS, AIIDALAB_REGISTRY
from .environment import AppEnvironment


def update_cache():
    """Run this process asynchronously."""
    requests_cache.install_cache(cache_name='apps_meta', backend='sqlite', expire_after=3600, old_data_on_error=True)
    requests.get(AIIDALAB_REGISTRY)
    requests_cache.install_cache(cache_name='apps_meta', backend='sqlite')


# Warning: try-except is a fix for Quantum Mobile release v19.03.0 that does not have requests_cache installed
try:
    import requests_cache
    # At start getting data from cache
    requests_cache.install_cache(cache_name='apps_meta', backend='sqlite')

    # If requests_cache is installed, upgrade the cache in the background.
    UPDATE_CACHE_BACKGROUND = bg.BackgroundJobFunc(update_cache)
    UPDATE_CACHE_BACKGROUND.start()
except ImportError:
    pass


def load_app_registry():
    """Load apps' information from the AiiDA lab registry."""
    parsed_url = urlparse(AIIDALAB_REGISTRY)
    if parsed_url.scheme == 'file':
        with open(parsed_url.path) as file:
            return json.loads(file.read())
    else:
        try:
            return requests.get(AIIDALAB_REGISTRY).json()
        except ValueError:
            print("Registry server is unavailable! Can't check for the updates")
            return dict(apps=dict(), catgories=dict())


def url_for(endpoint):
    """Generate the fully-qualified URL for a given endpoint."""
    parsed_url = urlsplit(endpoint)
    url_path = Path(parsed_url.path)
    query_string = parse_qs(parsed_url.query)

    try:
        app_name = url_path.resolve().relative_to(AIIDALAB_APPS).parts[0]
    except ValueError:
        raise ValueError("Endpoint argument to url_for() must be relative to the AIIDALAB_APPS path.")

    app_environment = AppEnvironment(app_name)

    if app_environment.installed():
        query_string.setdefault('kernel_name', app_environment.kernel_name)
    else:  # Use fallback kernel:
        query_string.setdefault('kernel_name', 'python3')

    return urlunsplit(parsed_url._replace(query=urlencode(query_string, doseq=True)))


class throttled:  # pylint: disable=invalid-name
    """Decorator to throttle calls to a function to a specified rate.

    The throttle is specific to the first argument of the wrapped
    function. That means for class methods it is specific to each
    instance.

    Adapted from: https://gist.github.com/gregburek/1441055

    """

    def __init__(self, calls_per_second=1):
        self.calls_per_second = calls_per_second
        self.last_start = defaultdict(lambda: -1)
        self.locks = defaultdict(Lock)

    def __call__(self, func):
        """Return decorator function."""

        @wraps(func)
        def wrapped(instance, *args, **kwargs):
            if self.last_start[hash(instance)] >= 0:
                elapsed = time.perf_counter() - self.last_start[hash(instance)]
                to_wait = 1.0 / self.calls_per_second - elapsed
                if to_wait > 0:
                    locked = self.locks[hash(instance)].acquire(blocking=False)
                    if locked:
                        try:
                            time.sleep(to_wait)
                        finally:
                            self.locks[hash(instance)].release()
                    else:
                        return None  # drop

            self.last_start[hash(instance)] = time.perf_counter()
            return func(instance, *args, **kwargs)

        return wrapped
