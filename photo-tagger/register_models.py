
import json
import pytoml
import os

from assets import Asset

entity_ref = {"entity_kind": "task", "entity_id": f"user:{os.environ['USER']}"}
asset = Asset(project='photo_tagger', branch='main', entity_ref=entity_ref)

for modelfile in os.listdir('models'):
    with open(f"models/{modelfile}") as f:
        config = pytoml.load(f)
        blobs = config['blobs']
        name = config['model']
        tags = config['tags']
        print('blobs', blobs)
        asset.register_model_asset(name, config['description'], config['kind'], blobs=blobs, tags=tags)
        print(f"Registered {name}")

#print("Current models:")
#for model in asset.list_model_assets():
#    print(json.dumps(model, indent=2))
