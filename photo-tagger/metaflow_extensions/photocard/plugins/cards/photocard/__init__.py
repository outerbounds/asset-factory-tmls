
import os
import json

from metaflow.plugins.cards.card_modules.card import MetaflowCard


ABS_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
GRID_TEMPLATE_PATH = os.path.join(ABS_DIR_PATH, "grid.html")
TAGS_TEMPLATE_PATH = os.path.join(ABS_DIR_PATH, "tags.html")
MODEL_TEMPLATE_PATH = os.path.join(ABS_DIR_PATH, "model.html")
EVALS_TEMPLATE_PATH = os.path.join(ABS_DIR_PATH, "evals.html")


class PhotoTaggerCard(MetaflowCard):

    type = 'phototagger'

    def __init__(self, options=None, **kwargs):
        pass

    def render(self, task):
        chevron = self._get_mustache()
        def_ldel='{{'

        replace = {}
        if 'photo_grid' in task:
            data = task['photo_grid'].data
            tmpl = GRID_TEMPLATE_PATH
        elif 'model_responses' in task:
            data = task['model_responses'].data
            tmpl = MODEL_TEMPLATE_PATH
        elif 'models_eval' in task:
            data = task['models_eval'].data
            tmpl = TAGS_TEMPLATE_PATH
        elif 'evals' in task:
            # this template contains embedded and minified javascript
            # which contains {{ }} sequence. Hence use a string
            # replacement instead 
            data = json.dumps(task['evals'].data)
            tmpl = EVALS_TEMPLATE_PATH
            replace = {'__CARD_JSON__': data}

        with open(tmpl) as f:
            if replace:
                txt = f.read()
                for k, v in replace.items():
                    txt = txt.replace(k, v)
                return txt
            else:
                return chevron.render(f, data)

CARDS = [PhotoTaggerCard]