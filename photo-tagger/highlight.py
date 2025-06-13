import base64, uuid
from functools import wraps
from metaflow import Flow, current

def mimetype(blob):
    if blob[6:10] in (b'JFIF', b'Exif'):
        return 'image/jpeg'
    elif blob[:4] == b'\xff\xd8\xff\xdb':
        return 'image/jpeg'
    elif blob.startswith(b'\211PNG\r\n\032\n'):
        return 'image/png'
    elif blob[:6] in (b'GIF87a', b'GIF89a'):
        return 'image/gif'
    else:
        return 'application/octet-stream'

class HighlightData():
    def __init__(self):
        self.title = ''
        self._labels = []
        self._synopsis = []
        self._columns = []
        self._image = None

    def set_title(self, title):
        self.title = title

    def add_label(self, label):
        self._labels.append(label)

    def add_line(self, line, caption=''):
        self._synopsis.append(f'**{caption}** {line}')

    def add_column(self, big='', small=''):
        self._columns.append({'big': big, 'small': small})

    def set_image(self, img):
        mime = mimetype(img)
        encoded = base64.b64encode(img).decode('ascii')
        self._image = f'data:{mime};base64,{encoded}'

    def _modified(self):
        return any(self._serialize().values())
    
    def _serialize(self):
        if self._image:
            htype = 'image'
            hbody = {'src': self._image}
        else:
            htype = 'columns'
            hbody = self._columns
        return {
            'type': 'highlight',
            'id': f'highlight-{uuid.uuid4()}',            
            'title': self.title,
            'labels': self._labels,
            'synopsis': self._synopsis,
            'highlight_type': htype,
            'highlight_body': hbody
        }

def highlight(f):

    @wraps(f)
    def wrapper(self, *args):
        self.highlight = HighlightData()
        try:
            if args:  
                print('inp', args)          
                f(self, args[0])
            else:
                f(self)
            if self.highlight._modified():
                Flow(current.flow_name)[current.run_id].add_tag("highlight")
        finally:
            self.highlight_data = self.highlight._serialize()
            del self.highlight

    from metaflow import card, ob_highlighter
    return card(type='highlight')(ob_highlighter(wrapper))
