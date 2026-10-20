"""Record the exact package versions of the Python environments used.

Writes ``envs/<env>.txt`` (one ``name==version`` per line; packages installed
from a local path or URL are listed with it). The environments are sb75's and
have neither pip nor uv, so ``importlib.metadata`` of each interpreter is used.

    python envs/record_environments.py      (any Python >= 3.8)
"""
import subprocess
import sys
from pathlib import Path

ENVS = {
    'banksy': '/nfs/team361/sb75/.venvs/banksy/bin/python',
    'cellcharter': '/nfs/team361/sb75/.venvs/cellcharter/bin/python',
    'nichecompass': '/nfs/team361/sb75/.venvs/nichecompass/bin/python',
    'terra': '/nfs/team361/sb75/.venvs/terra/bin/python',
    'novae': '/nfs/team361/sb75/.venvs/novae/bin/python',  # checks/test_novae_correction.py
}

LIST_PACKAGES = r'''
import json, sys
from importlib.metadata import distributions
print(f"# python {sys.version.split()[0]} ({sys.executable})")
for d in sorted(distributions(), key=lambda d: (d.metadata["Name"] or "").lower()):
    name, version = d.metadata["Name"], d.version
    source = ""
    try:
        info = json.loads(d.read_text("direct_url.json") or "{}")
        source = info.get("url", "")
    except Exception:
        pass
    print(f"{name}=={version}" + (f"  # {source}" if source else ""))
'''

out_dir = Path(__file__).resolve().parent
for env, python in ENVS.items():
    result = subprocess.run([python, '-c', LIST_PACKAGES], capture_output=True,
                            text=True, check=True)
    (out_dir / f'{env}.txt').write_text(result.stdout)
    n = sum(1 for line in result.stdout.splitlines() if '==' in line)
    print(f'{env}: {n} packages -> envs/{env}.txt')
sys.exit(0)
