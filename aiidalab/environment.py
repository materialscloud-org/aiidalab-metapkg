# -*- coding: utf-8 -*-
"""Manage Python environments and Jupyter kernels for apps."""
import re
import venv
import shutil
from hashlib import sha1
from pathlib import Path
from subprocess import run
from urllib.parse import quote_plus

from jupyter_core.paths import jupyter_data_dir

from .config import AIIDALAB_APPS


def _valid_jupyter_kernel_name(name):
    """Check whether the given name is a valid Jupyter kernel name."""
    return re.fullmatch(r'[a-z0-9\-\_.]+', name)


class AppEnvironmentError(RuntimeError):
    """Indicates issue relating to the AppEnvironment installation."""


class AppEnvironment:
    """Manage Python environment and Jupyter kernel for AiiDA lab app.

    Arguments:

        app_name (str):
            Name of the AiiDA lab app.
    """

    def __init__(self, app_name):
        self._app_name = app_name

    @property
    def kernel_name(self):
        """Generate unique name for this kernel based on the app name.

        The kernel name is guaranteed to be a valid Jupyter kernel name
        and unique with respect to the app name so to avoid kernel name
        collisions between different apps.
        """
        # The unique_name is a hash value based on the app_name which is
        # guaranteed to be truly unique with respect to the app name.
        # The current schema version (1) is explicitly encoded to simplify
        # potential future migrations.
        unique_name = sha1(f'1/{self._app_name}'.encode()).hexdigest()

        # The human-readable name is a name based on the app name that
        # constitutes a valid Jupyter kernel name:
        human_readable_name = quote_plus(self._app_name.replace('@', '.')).replace('%', '-')
        kernel_name = f'{human_readable_name:.32}-{unique_name:.8}'

        # Return only the unique_name in case that we failed to
        # construct a valid kernel name:
        if not _valid_jupyter_kernel_name(kernel_name):
            return unique_name

        return kernel_name

    @property
    def prefix(self):
        return Path(AIIDALAB_APPS).joinpath(self._app_name, '.venv')

    @property
    def executable(self):
        return self.prefix.joinpath('bin', 'python')

    @property
    def jupyter_kernel_path(self):
        return Path(jupyter_data_dir()).joinpath('kernels', self.kernel_name)

    def install(self, system_site_packages=True, clear=True):
        """Create the Python virtual environment and install the Jupyter kernel."""
        venv.create(self.prefix, system_site_packages=system_site_packages, clear=clear)
        run([self.executable, '-c', 'import reentry; reentry.manager.scan()'], check=True)
        run([self.executable, '-m', 'ipykernel', 'install', '--user', f'--name={self.kernel_name}'], check=True)

    def uninstall(self):
        """Remove both the Python virtual environment and the corresponding Jupyter kernel."""
        try:
            shutil.rmtree(self.jupyter_kernel_path)
        except FileNotFoundError:
            pass  # kernel was not installed
        try:
            shutil.rmtree(self.prefix)
        except FileNotFoundError:
            pass  # environment was not installed

    def installed(self):
        """Check whether this environment is (properly) installed.

        Returns True if environment is installed, otherwise False.

        Raises AppEnvironmentError exception if the installation state
        is inconistent.
        """
        # First, check the consistency of the installation state:
        if self.prefix.is_dir():

            if not self.executable.is_file():
                raise AppEnvironmentError(f"The environment executable ('{self.executable}') is missing.")

            if not self.jupyter_kernel_path.is_dir():
                raise AppEnvironmentError(
                    f"The app has an app-specific virtual environment ('{self.prefix}'), "
                    f"but the corresponding Jupyter kernel ('{self.jupyter_kernel_path}') is not installed.")

        if self.jupyter_kernel_path.is_dir():

            if not self.prefix.is_dir():
                raise AppEnvironmentError(f"The kernel for this app ({self.jupyter_kernel_path}) is installed, "
                                          f"but the corresponding virtual environment ({self.prefix}) does not exist.")

        # Passed consistency check, return installation state:
        return self.prefix.is_dir() and self.jupyter_kernel_path.is_dir()
