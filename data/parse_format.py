import json

def convert_jsonl_input(dir, file):
    collection = {'collection': []}
    with open(f'data/{dir}/{file}.jsonl') as f:
        data = [json.loads(line) for line in f.readlines()]
    covered_dialogues = set()
    for d in data:
        did = d["id"]
        if did not in covered_dialogues:
            trunc_context = d["input"][:d["input"].find('[Question]')]
            turns = [
                x.replace('Speaker: ', '').replace('Listener: ', '') 
                for x in trunc_context.strip().split('\n')
            ]
            dialogue = {
                'dialogue_id': did, 
                'turns': [{
                    'uid': j,
                    'sid': 'S1' if j % 2 == 0 else 'S2',
                    'utt': t
                } for j, t in enumerate(turns)],
                'on_terminal': True
            }
            collection['collection'].append(dialogue)
            covered_dialogues.add(did)
    json.dump(collection, open(f'data/{dir}/{file}_onlydialogues.json', 'w'), indent=2)


if __name__ == '__main__':
    convert_jsonl_input('convosense', 'train')
    convert_jsonl_input('convosense', 'dev')
    convert_jsonl_input('convosense', 'test')