"""Read-only SHA-anchored proof relocation; frozen protocol bytes are unchanged."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def regular_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Proof must be an existing regular file: " + str(path))
    return path.resolve()


class ProofPaths:
    """Verify every override against its original frozen SHA before any read.

    Import paths are aliases, never changes to the GPU protocol or native model
    configuration. Additional local evidence must be an exact imported file.
    """
    def __init__(self, output, spec, imported=None):
        self.output = Path(output).resolve()
        self.spec = spec
        manifest_path = self.output / "local_import_manifest.json"
        self.imported = read_json(manifest_path) if imported is None and manifest_path.exists() else imported
        self.expected = {}
        for entry in spec["pinned_files"]:
            if entry["path"] in self.expected:
                raise ValueError("Pinned proof paths must be unique")
            self.expected[entry["path"]] = entry["sha256"]
        self.expected[spec["model_provenance_path"]] = spec["model_provenance_sha256"]
        self.expected[spec["backend_evidence_path"]] = spec["backend_evidence_sha256"]
        self.overrides, self.imported_files = {}, {}
        if self.imported is None:
            return
        if (self.imported["gpu_protocol_sha256"] != sha256_file(self.output / "frozen_spec.json")
                or self.imported.get("gpu_original_spec_and_contracts_unchanged") is not True
                or self.imported.get("prediction_values_changed") is not False
                or self.imported.get("cpu_gpu_prediction_equality_claimed") is not False):
            raise AssertionError("Imported evidence is not bound to this unchanged GPU protocol")
        for item in self.imported["files"]:
            relative = PurePosixPath(item["path"])
            if relative.is_absolute() or ".." in relative.parts or "\\" in item["path"]:
                raise ValueError("Imported proof file paths must be normalized relative paths")
            name = relative.as_posix()
            if name in self.imported_files:
                raise ValueError("Imported proof paths must be unique")
            self.imported_files[name] = item["sha256"]
        if self.imported_files.get("frozen_spec.json") != self.imported["gpu_protocol_sha256"]:
            raise AssertionError("The imported file list does not anchor the GPU protocol")
        self.overrides = dict(self.imported["proof_path_overrides"])
        if set(self.overrides) != set(self.expected):
            raise AssertionError("Every override must correspond exactly to frozen source, weight, provenance or backend evidence")
        for remote, local in self.overrides.items():
            checked = regular_file(local)
            if sha256_file(checked) != self.expected[remote]:
                raise AssertionError("A relocated proof differs from its frozen SHA256: " + remote)

    def resolve(self, path, expected_sha256=None):
        key = str(path)
        expected = self.expected.get(key) if expected_sha256 is None else expected_sha256
        if key in self.expected and expected != self.expected[key]:
            raise AssertionError("An explicit proof SHA disagrees with the frozen protocol")
        if expected is None:
            raise ValueError("Proof reads require an independently declared SHA256")
        selected = regular_file(self.overrides.get(key, path))
        if sha256_file(selected) != expected:
            raise AssertionError("Proof bytes changed: " + key)
        return selected

    def local_evidence(self, relative):
        relative = PurePosixPath(str(relative))
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
            raise ValueError("Local evidence must stay inside the experiment")
        path = self.output.joinpath(*relative.parts)
        selected = regular_file(path)
        if not selected.is_relative_to(self.output):
            raise ValueError("Local evidence escaped its imported experiment")
        if self.imported is not None:
            expected = self.imported_files.get(relative.as_posix())
            if expected is None or sha256_file(selected) != expected:
                raise AssertionError("Local evidence is missing from or differs from the imported SHA list")
        return selected


def resolve_proof_path(path, output, spec, imported=None, expected_sha256=None):
    """Convenience API; prefer one ProofPaths instance for multiple files."""
    return ProofPaths(output, spec, imported).resolve(path, expected_sha256)
