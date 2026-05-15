from __future__ import annotations

import mimetypes
import sys

_ORIG_GUESS_TYPE = mimetypes.guess_type

_STATIC_MIME_OVERRIDES: tuple[tuple[str, str], ...] = (
    (".js", "application/javascript"),
    (".mjs", "application/javascript"),
    (".css", "text/css"),
    (".json", "application/json"),
    (".svg", "image/svg+xml"),
    (".ico", "image/x-icon"),
    (".woff2", "font/woff2"),
    (".map", "application/json"),
)


def _guess_type_fixed(url: str, strict: bool = True) -> tuple[str | None, str | None]:
    path_no_query = url.split("?", 1)[0]
    lower_path = path_no_query.lower()
    for ext, ctype in _STATIC_MIME_OVERRIDES:
        if lower_path.endswith(ext):
            _, enc = _ORIG_GUESS_TYPE(url, strict)
            return ctype, enc
    return _ORIG_GUESS_TYPE(url, strict)


for ext, ctype in _STATIC_MIME_OVERRIDES:
    mimetypes.add_type(ctype, ext, strict=True)
    mimetypes.add_type(ctype, ext, strict=False)

mimetypes.guess_type = _guess_type_fixed


def main() -> None:
    rest = sys.argv[1:]
    if not rest:
        rest = ["ui", "--host", "127.0.0.1", "--port", "5000"]
    sys.argv = ["mlflow", *rest]

    from mlflow.cli import cli

    cli.main()


if __name__ == "__main__":
    main()
