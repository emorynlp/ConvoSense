import os, json, pickle
from datetime import datetime
import cattrs
from data.dialogues_struct import Turn, Dialogue, Dialogues


def load_data(file=None) -> Dialogues:
    if file is None:
        file = 'dialogues.json'
    if file.endswith('.json'):
        raw_data = json.load(open(f'{os.path.dirname(__file__)}/{file}'))
        if 'collection' not in raw_data:
            raw_data = {'collection': raw_data}
        items = cattrs.structure(raw_data, Dialogues)
    return items


def save_data(data, file):
    data_to_dump = cattrs.unstructure(data)
    json.dump(
        data_to_dump,
        open(f'{os.path.dirname(__file__)}/{file}', 'w'),
        indent=2
    )


def rebuild_and_save_cs(data, file=None):
    if file is None:
        file = 'dialogues-cs.json'
    save_data(data, file)


def rebuild_and_save_response(data, file=None):
    if file is None:
        file = f'dialogues-response-{datetime.now().strftime("%m%d%Y-%H%M%S")}.json'
    save_data(data, file)