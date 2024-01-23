import os, json, pickle
from datetime import datetime
import cattrs
from data.dialogues_struct import Turn, Dialogue, Dialogues


def load_data(filepath) -> Dialogues:
    raw_data = json.load(open(filepath))
    if 'collection' not in raw_data:
        raw_data = {'collection': raw_data}
    items = cattrs.structure(raw_data, Dialogues)
    return items


def save_data(data, filepath):
    data_to_dump = cattrs.unstructure(data)
    json.dump(
        data_to_dump,
        open(filepath, 'w'),
        indent=2
    )


def rebuild_and_save_cs(data, filepath):
    save_data(data, filepath)


def rebuild_and_save_response(data, filepath):
    save_data(data, filepath)