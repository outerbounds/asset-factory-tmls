import sys
import json
from metaflow import current
from assets import Asset

MAGIC_PREFIX = 'usm0jrb3'

class EvalsLogger():

    def __init__(self, project=None, branch=None):
        asset = Asset(project=project, branch=branch)
        self.meta = asset.meta

    def log(self, msg):
        jsonmsg = json.dumps(self.meta | {'content': msg})
        print(f"{MAGIC_PREFIX} evals {jsonmsg}")
        #sys.stdout.flush()
