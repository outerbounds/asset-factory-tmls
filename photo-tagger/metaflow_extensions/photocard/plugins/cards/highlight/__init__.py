import os
import json
import traceback

from metaflow.plugins.cards.card_modules.card import MetaflowCard

ABS_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(ABS_DIR_PATH, "base.html")

class HighlightCard(MetaflowCard):
    ALLOW_USER_COMPONENTS = True
    RUNTIME_UPDATABLE = False

    type = "highlight"

    def render(self, task):

        chevron = self._get_mustache()

        if 'highlight_data' in task:
            spec = task['highlight_data'].data
        else:
            spec = {}
        with open(TEMPLATE_PATH) as f:
            return chevron.render(f, {'highlight_spec': spec})

CARDS = [HighlightCard]
