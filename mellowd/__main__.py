import sys

if "--package-check" in sys.argv:
    from mellowd.packagecheck import run
else:
    from mellowd.main import run

run()
