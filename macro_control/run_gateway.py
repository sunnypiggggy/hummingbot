from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("DCA_MACRO_BIND_HOST", "127.0.0.1")
    container_bind_allowed = (
        os.environ.get("DCA_MACRO_ALLOW_CONTAINER_BIND", "false").lower()
        == "true"
    )
    if host not in {"127.0.0.1", "::1", "localhost"} and not (
        host == "0.0.0.0" and container_bind_allowed
    ):
        raise RuntimeError(
            "non-loopback bind requires DCA_MACRO_ALLOW_CONTAINER_BIND=true"
        )
    uvicorn.run(
        "macro_control.app:app_from_environment",
        factory=True,
        host=host,
        port=int(os.environ.get("DCA_MACRO_BIND_PORT", "8791")),
        access_log=False,
    )


if __name__ == "__main__":
    main()
