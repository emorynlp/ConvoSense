from models.best_model import GPT_CS_GENERATOR
from models.utils import COMMONSENSE_LS
import data.load as data_loader


if __name__ == '__main__':
    data = data_loader.load_data(
        filepath='data/convosense/test_onlydialogues.json'
    )
        
    model = GPT_CS_GENERATOR
    model.load_model(model.modelpath)
    model.attach()

    items_with_q = []
    for dialogue in data[:1]:
        for turn in dialogue.turns_to_execute():
            for q in COMMONSENSE_LS:
                item = {
                    'id': dialogue.dialogue_id,
                    'uid': turn.uid,
                    'turn_obj': turn,
                    'context': dialogue.context(turn.uid),
                    'cs': turn.cs,
                    'turns': [t.utt for t in dialogue.turns[:turn.uid+1]],
                    'question': q
                }
                items_with_q.append(item)

    formatted_data = model.format_data(data=items_with_q)
    model.generate(data=formatted_data)

    for result in formatted_data:
        turn = result['turn_obj']
        # turn.cs[result['question']] = result['beamed_generations'][0]         # -> to save 1 output only
        turn.beam_cs[result['question']] = result['beamed_generations'][:5]     # -> to save K outputs

    data_loader.rebuild_and_save_cs(
        data=data, 
        filepath=f'gen/{model.name}_convosense_test.json'
    )