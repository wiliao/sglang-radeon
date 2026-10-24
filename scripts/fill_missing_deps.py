#!/usr/bin/env python3
"""Install the second-order requirements that ``--no-deps`` leaves behind.

``install.sh`` installs SGLang, the plugin and the dependency list with
``--no-deps`` so that pip's resolver can never swap the ROCm torch for a CUDA
build. The cost is that nothing resolves *transitive* requirements either. On a
fresh virtualenv that is not a theoretical problem:

    import sglang
    ModuleNotFoundError: No module named 'urllib3'

and even the plugin's own metadata build fails, because ``huggingface_hub``
imports ``httpx``:

    ModuleNotFoundError: No module named 'httpx'
      [end of output]
    error: metadata-generation-failed

``install.sh`` already hand-adds three such packages (huggingface_hub,
apache-tvm-ffi, kernels_data). That list cannot stay complete: one fresh venv
here was missing 69 distributions, including urllib3, httpx, certifi, idna,
pydantic-core, tokenizers and safetensors.

Instead of maintaining the list by hand, derive it from what is installed. Walk
every distribution's metadata, find requirements whose marker applies and whose
distribution is absent, and install exactly those. Repeat, because installing
one can add new requirements.

Nothing that could replace the caller's chosen torch is ever installed -- see
FORBIDDEN. Those versions are decisions install.sh made deliberately, and a
"helpful" reinstall is how a ROCm box ends up on a CUDA torch with no error.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import re
import subprocess
import sys

from packaging.requirements import Requirement

#: Never an argument to pip here. Either the caller pinned it (torch, triton),
#: it is CUDA-only (cuda-*, flashinfer*, nvidia-*), or it is built from source
#: on purpose (sglang-kernel / sglang-kernel-rdna3).
FORBIDDEN = re.compile(
    r"^(torch|torchvision|torchaudio|triton|triton-rocm|torchcodec|"
    r"torch_memory_saver|torch-c-dlpack-ext|"
    r"cuda.*|nvidia.*|flashinfer.*|"
    r"sglang|sglang-kernel|sglang-kernel-rdna3|sglang-radeon-rdna3|"
    r"humming.*|quack.*|kernels-data|sgl-amd-mono)$",
    re.I,
)


def _installed(name: str) -> bool:
    try:
        md.version(name)
    except md.PackageNotFoundError:
        return False
    return True


def missing() -> dict[str, str]:
    """Requirements that apply to this interpreter but are not installed."""
    found: dict[str, str] = {}
    for dist in md.distributions():
        for raw in dist.requires or []:
            try:
                req = Requirement(raw)
            except Exception:
                continue  # malformed metadata is not our problem
            if req.marker is not None:
                try:
                    if not req.marker.evaluate():
                        continue
                except Exception:
                    continue
            if FORBIDDEN.match(req.name) or _installed(req.name):
                continue
            spec = f"{req.name}{req.specifier}" if req.specifier else req.name
            found.setdefault(req.name.lower(), spec)
    return found


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-rounds", type=int, default=6,
                    help="install rounds before giving up (default: 6)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be installed and exit")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    total = 0
    for rnd in range(args.max_rounds):
        todo = sorted(missing().values())
        if not todo:
            if not args.quiet:
                print(f"second-order requirements complete "
                      f"({total} added over {rnd} round(s))")
            return 0
        if args.dry_run:
            print(" ".join(todo))
            return 0
        if not args.quiet:
            print(f"  round {rnd}: {len(todo)} missing "
                  f"({', '.join(todo[:6])}{'...' if len(todo) > 6 else ''})")
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--progress-bar", "off",
             "--no-input", *todo],
            check=False,
        )
        total += len(todo)
        if proc.returncode != 0:
            print(f"warning: pip exited {proc.returncode} installing "
                  f"second-order requirements", file=sys.stderr)
            return 0  # not fatal: the caller's doctor decides
    if not args.quiet:
        print("warning: still missing requirements after "
              f"{args.max_rounds} rounds", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
