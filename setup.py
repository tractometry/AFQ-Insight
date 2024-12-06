import os.path as op

from setuptools import setup


def local_version(version):
    """Patch in a version that can be uploaded to test PyPI."""
    return ""


opts = {
    "use_scm_version": {
        "write_to": op.join("afqinsight", "_version.py"),
        "local_scheme": local_version,
    }
}


if __name__ == "__main__":
    setup(**opts)
