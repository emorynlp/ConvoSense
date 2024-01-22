import json

def load_gpt_test_set():
    filepath = f'data/convosense/test.jsonl'
    items = [json.loads(item) for item in open(filepath).readlines()]
    nonduplicates = []
    covered = set()
    for item in items:
        key = f'{item["id"]}_{item["type"]}'
        if key not in covered:
            nonduplicates.append(item)
            covered.add(key)
    return filepath, nonduplicates

def load_human_test_set():
    filepath = f'data/humangen/test.jsonl'
    items = [json.loads(item) for item in open(filepath).readlines()]
    nonduplicates = []
    covered = set()
    for item in items:
        key = f'{item["id"]}_{item["context"]}_{item["question"]}'
        if key not in covered:
            nonduplicates.append(item)
            covered.add(key)
    return filepath, nonduplicates