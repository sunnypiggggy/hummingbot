"""Create local Hermes mTLS material and a rotatable HMAC keyring."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
from pathlib import Path


def run(*args: str) -> None:
    subprocess.run(["openssl", *args], check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    files = {
        "ca_key": output / "hermes-client-ca.key",
        "ca_cert": output / "hermes-client-ca.pem",
        "client_key": output / "hermes-client.key",
        "client_csr": output / "hermes-client.csr",
        "client_cert": output / "hermes-client.pem",
        "extensions": output / "client-extensions.cnf",
        "openssl_config": output / "openssl.cnf",
        "hmac": output / "hermes_hmac_secrets.json",
    }
    existing = [path for path in files.values() if path.exists()]
    if existing:
        raise RuntimeError(
            "refusing to overwrite existing credentials: "
            + ", ".join(str(path) for path in existing)
        )

    run(
        "ecparam",
        "-name",
        "prime256v1",
        "-genkey",
        "-noout",
        "-out",
        str(files["ca_key"]),
    )
    files["openssl_config"].write_text(
        "[req]\n"
        "distinguished_name=distinguished_name\n"
        "prompt=no\n"
        "[distinguished_name]\n",
        encoding="utf-8",
    )
    run(
        "req",
        "-x509",
        "-new",
        "-sha256",
        "-key",
        str(files["ca_key"]),
        "-out",
        str(files["ca_cert"]),
        "-days",
        str(args.days),
        "-config",
        str(files["openssl_config"]),
        "-addext",
        "basicConstraints=critical,CA:TRUE",
        "-addext",
        "keyUsage=critical,keyCertSign,cRLSign",
        "-subj",
        "/CN=Hermes DCA Client CA",
    )
    run(
        "ecparam",
        "-name",
        "prime256v1",
        "-genkey",
        "-noout",
        "-out",
        str(files["client_key"]),
    )
    run(
        "req",
        "-new",
        "-sha256",
        "-key",
        str(files["client_key"]),
        "-out",
        str(files["client_csr"]),
        "-config",
        str(files["openssl_config"]),
        "-subj",
        "/CN=hermes-macro",
    )
    files["extensions"].write_text(
        "basicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature,keyAgreement\n"
        "extendedKeyUsage=critical,clientAuth\n",
        encoding="utf-8",
    )
    run(
        "x509",
        "-req",
        "-sha256",
        "-in",
        str(files["client_csr"]),
        "-CA",
        str(files["ca_cert"]),
        "-CAkey",
        str(files["ca_key"]),
        "-CAcreateserial",
        "-out",
        str(files["client_cert"]),
        "-days",
        str(args.days),
        "-extfile",
        str(files["extensions"]),
    )
    files["hmac"].write_text(
        json.dumps(
            {
                "primary": secrets.token_urlsafe(48),
                "next": secrets.token_urlsafe(48),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for name in (
        "ca_key",
        "client_key",
        "client_csr",
        "extensions",
        "openssl_config",
        "hmac",
    ):
        try:
            os.chmod(files[name], 0o600)
        except OSError:
            pass
    print(f"credentials created in {output}")
    print("copy only hermes-client-ca.pem and the HMAC JSON to OCI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
