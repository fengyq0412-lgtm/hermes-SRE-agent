"""Run one argv command against a disposable copy of the filtered project."""
import json
import os
import shutil
import subprocess


shutil.copytree("/source", "/tmp/project")
os.chdir("/tmp/project")
argv = json.loads(os.environ["HERMES_COMMAND_JSON"])
environment = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
               "HOME": "/tmp", "LANG": "C.UTF-8", "PYTHONPATH": "/tmp/project:/tmp/project/src"}
raise SystemExit(subprocess.call(argv, env=environment))
