"""Smoke test: execute every code cell of each notebook in solutions/ (no Jupyter needed)."""
import json, sys, os, io, contextlib
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
folder = sys.argv[1] if len(sys.argv) > 1 else "solutions"
ok = True
for nb in sorted((ROOT / folder).glob("*.ipynb")):
    os.chdir(ROOT / folder)
    ns, buf = {"__name__": "__main__"}, io.StringIO()
    try:
        for c in json.loads(nb.read_text())["cells"]:
            if c["cell_type"] == "code":
                with contextlib.redirect_stdout(buf):
                    exec("".join(c["source"]), ns)
        print("PASS", nb.name)
    except Exception as e:
        ok = False; print("FAIL", nb.name, repr(e)); print(buf.getvalue()[-800:])
sys.exit(0 if ok else 1)
