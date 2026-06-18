"""One command to keep the site in sync after adding a run or editing cells:

    bake cell protocols  →  refit the 4-D model  →  re-embed it  →  validate

Run this after editing reproductions/ or the cell arrays in site/explore.html so
the protocol families and the prediction model never drift.

    python scripts/refit_site.py
"""

import re
import subprocess
import sys

PY = sys.executable


def run(*a):
    subprocess.run([PY, *a], check=True)


def main():
    print("• baking cell protocols (single source of truth)…")
    run("scripts/site_protocols.py", "--write")

    print("• fitting the 4-D model (protocol × method × dataset × metric, LOO-calibrated)…")
    model = subprocess.run([PY, "scripts/bo_table_forecast.py", "--model"],
                           check=True, capture_output=True, text=True).stdout.strip()

    print("• re-embedding the model into the site…")
    h = open("site/explore.html").read()
    h2 = re.sub(r'(<script type="application/json" id="model-fit">\n).*?(\n  </script>)',
                lambda m: m.group(1) + model + m.group(2), h, flags=re.S)
    if h2 == h:
        sys.exit("ERROR: model-fit block not found in site/explore.html")
    open("site/explore.html", "w").write(h2)

    print("• validating…")
    run("-m", "pytest", "tests/test_site_coverage.py", "-q")
    print("✅ site refit complete — protocols baked, model embedded, coverage consistent")


if __name__ == "__main__":
    main()
