"""Record what went into and came out of a model run, so the run can be checked or reproduced later.

A run manifest (``run_manifest.json`` in the run directory) holds the SHA-256 hash and size of every file under the
run's ``input`` and ``output`` directories, the hashes of the settings files, the model data directory, and the
versions and ``OASIS_*`` environment variables the run used.

Two runs whose manifests list the same file hashes produced byte-identical inputs and outputs.
"""

__all__ = ['MANIFEST_NAME', 'build_manifest', 'write_manifest', 'read_manifest', 'compare_manifest', 'verify_run']

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata

from .exceptions import OasisException

MANIFEST_NAME = 'run_manifest.json'
MANIFEST_VERSION = 1
HASHED_DIRS = ('input', 'output')
PACKAGES = ('oasislmf', 'ods-tools', 'oasis-data-manager', 'numpy', 'numba', 'pandas', 'pyarrow', 'scipy')
CHUNK_SIZE = 1 << 20


def file_sha256(path):
    """Return the hex SHA-256 digest of a file.

    Args:
        path (str): File path.

    Returns:
        str: The digest.
    """
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b''):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root, skip=()):
    """Hash every regular file under a directory, following symbolic links to files.

    Args:
        root (str): Directory to hash.
        skip (iterable): Paths, relative to ``root`` with ``/`` separators, to leave out.

    Returns:
        dict: ``{relative path: {'sha256': ..., 'size': ...}}``, sorted by path.
    """
    skip = set(skip)
    files = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in filenames:
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root).replace(os.sep, '/')
            if rel in skip or not os.path.isfile(path):
                continue
            files[rel] = {'sha256': file_sha256(path), 'size': os.path.getsize(path)}
    return dict(sorted(files.items()))


def _package_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_info():
    """Return the Python, platform and package versions and the ``OASIS_*`` environment variables.

    Returns:
        dict: Environment details.
    """
    return {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'machine': platform.machine(),
        'packages': _package_versions(),
        'oasis_env': {k: v for k, v in sorted(os.environ.items()) if k.startswith('OASIS_')},
    }


def build_manifest(run_dir, settings_files=None, model_data_dir=None, hash_model_data=False):
    """Build the manifest of a finished run.

    Args:
        run_dir (str): Model run directory.
        settings_files (dict): ``{name: path}`` of settings files to hash, such as the analysis settings. Missing or
            None paths are left out.
        model_data_dir (str): Model data directory. Defaults to the run's ``static`` directory, which holds the model
            data files the run used (or links to them).
        hash_model_data (bool): Also hash every file in the model data directory. This can take a while for large
            models.

    Returns:
        dict: The manifest.
    """
    if not os.path.isdir(run_dir):
        raise OasisException(f"run directory '{run_dir}' not found")
    files = {}
    for name in HASHED_DIRS:
        sub = os.path.join(run_dir, name)
        if os.path.isdir(sub):
            skip = {MANIFEST_NAME}
            files.update({f'{name}/{rel}': info for rel, info in hash_tree(sub, skip).items()})

    static = os.path.join(run_dir, 'static')
    if model_data_dir is None and os.path.isdir(static):
        # A run's static directory is a link to the model data, or a directory of links to (or copies of) the files used
        model_data_dir = os.path.realpath(static) if os.path.islink(static) else static
    model_data = None
    if model_data_dir:
        model_data = {'path': os.path.abspath(model_data_dir), 'files': None}
        if hash_model_data:
            model_data['files'] = hash_tree(model_data_dir)

    settings = {}
    for name, path in (settings_files or {}).items():
        if path and os.path.isfile(path):
            settings[name] = {'path': os.path.abspath(path), 'sha256': file_sha256(path)}

    return {
        'manifest_version': MANIFEST_VERSION,
        'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'run_dir': os.path.abspath(run_dir),
        'environment': environment_info(),
        'settings': settings,
        'model_data': model_data,
        'files': files,
    }


def write_manifest(run_dir, manifest_fp=None, **kwargs):
    """Build the manifest of a run and write it to ``manifest_fp`` (default ``<run_dir>/run_manifest.json``).

    Args:
        run_dir (str): Model run directory.
        manifest_fp (str): Where to write the manifest.
        **kwargs: Passed to :func:`build_manifest`.

    Returns:
        dict: The manifest written.
    """
    manifest = build_manifest(run_dir, **kwargs)
    with open(manifest_fp or os.path.join(run_dir, MANIFEST_NAME), 'w') as f:
        json.dump(manifest, f, indent=2)
    return manifest


def read_manifest(path):
    """Read a manifest, given its path or the run directory that holds it.

    Args:
        path (str): ``run_manifest.json`` path, or a run directory.

    Returns:
        dict: The manifest.
    """
    if os.path.isdir(path):
        path = os.path.join(path, MANIFEST_NAME)
    if not os.path.isfile(path):
        raise OasisException(f"run manifest '{path}' not found; write one with 'oasislmf model manifest' "
                             "or 'oasislmf model run --run-manifest'")
    with open(path) as f:
        manifest = json.load(f)
    if manifest.get('manifest_version') != MANIFEST_VERSION:
        raise OasisException(f"'{path}' has manifest version {manifest.get('manifest_version')!r}; "
                             f"this oasislmf reads version {MANIFEST_VERSION}")
    return manifest


def _diff_files(expected, actual):
    shared = expected.keys() & actual.keys()
    return {
        'changed': sorted(p for p in shared if expected[p]['sha256'] != actual[p]['sha256']),
        'missing': sorted(expected.keys() - actual.keys()),
        'added': sorted(actual.keys() - expected.keys()),
        'matched': len(shared) - sum(expected[p]['sha256'] != actual[p]['sha256'] for p in shared),
    }


def _diff_flat(expected, actual, prefix=''):
    diffs = {}
    for key in sorted(set(expected) | set(actual)):
        a, b = expected.get(key), actual.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            diffs.update(_diff_flat(a, b, f'{prefix}{key}.'))
        elif a != b:
            diffs[f'{prefix}{key}'] = {'expected': a, 'actual': b}
    return diffs


def compare_manifest(expected, actual):
    """Compare two manifests.

    Args:
        expected (dict): The reference manifest.
        actual (dict): The manifest to check against it.

    Returns:
        dict: ``files`` (lists of ``changed``, ``missing`` and ``added`` paths, and the ``matched`` count),
        ``settings`` and ``model_data`` (paths whose hash differs), ``environment`` (differing entries) and ``match``
        (True when no input, output, settings or model data file differs).
    """
    files = _diff_files(expected['files'], actual['files'])
    settings = sorted(name for name in set(expected['settings']) | set(actual['settings'])
                      if expected['settings'].get(name, {}).get('sha256') != actual['settings'].get(name, {}).get('sha256'))
    model_data = []
    expected_md = (expected.get('model_data') or {}).get('files')
    actual_md = (actual.get('model_data') or {}).get('files')
    if expected_md is not None and actual_md is not None:
        md = _diff_files(expected_md, actual_md)
        model_data = md['changed'] + md['missing'] + md['added']
    match = not (files['changed'] or files['missing'] or files['added'] or settings or model_data)
    return {
        'match': match,
        'files': files,
        'settings': settings,
        'model_data': model_data,
        'environment': _diff_flat(expected['environment'], actual['environment']),
    }


def verify_run(run_dir, against=None):
    """Check a run's files against its own manifest, or against another run's.

    Against its own manifest, the run's files, settings files and (if the manifest hashed them) model data files are
    hashed again and compared, and the environment is compared with the current one.

    Against another run, the run's files are hashed and compared, and the model data files too if both manifests
    hashed them. The settings and environment come from the run's own manifest, or, without one, the settings are not
    compared and the environment is the current one. The run's ``input`` directory holds a copy of the analysis
    settings, so a changed analysis shows up as a changed file either way.

    Args:
        run_dir (str): Model run directory to check.
        against (str): Manifest path, or run directory, to check against. Defaults to ``run_dir``'s own manifest.

    Returns:
        dict: Result of :func:`compare_manifest`.
    """
    expected = read_manifest(against or run_dir)
    expected_md = expected.get('model_data') or {}
    hash_model_data = expected_md.get('files') is not None
    if not against:
        actual = build_manifest(
            run_dir,
            settings_files={name: info['path'] for name, info in expected['settings'].items()},
            model_data_dir=expected_md.get('path'),
            hash_model_data=hash_model_data,
        )
        return compare_manifest(expected, actual)

    own_fp = os.path.join(run_dir, MANIFEST_NAME)
    own = read_manifest(own_fp) if os.path.isfile(own_fp) else None
    own_md = (own or {}).get('model_data') or {}
    actual = build_manifest(run_dir, model_data_dir=own_md.get('path'),
                            hash_model_data=hash_model_data and own_md.get('files') is not None)
    if own is not None:
        actual['settings'], actual['environment'] = own['settings'], own['environment']
    else:
        actual['settings'] = expected['settings']
    return compare_manifest(expected, actual)
