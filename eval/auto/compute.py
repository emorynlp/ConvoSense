import transformers, random
random.seed(1234)
transformers.set_seed(1234)
from models.best_model import *
from eval.auto.utils import (
    load_results,
    save_results,
    format_data, 
    SELECTED_ARGS, 
    BEAMED_GENERATIONS, 
    load_t5, 
    DataCollatorForT5, 
    get_generations, 
)
from eval.auto.load_data import *
from eval.auto.auto_metrics import bertscore, sacrebleu, sentencebert
from models.best_model import GPT_CS_GENERATOR
import re, sys
import pandas as pd

def convert_data_format_per_model(formatted_data, modelname):
    for d in formatted_data:
        if 'soda' in modelname and 'source' in d and ('comfact' in d['source'] or 'cicero' in d['source'] or 'reflect' in d['source']):
            d['input'] = re.sub('Speaker (.+):', 'Speaker:', d['input'])
            d['input'] = re.sub('Listener (.+):', 'Listener:', d['input'])
        elif 'human' in modelname and ('source' not in d or d['source'] == 'None'):
            speaker_idx = d['input'].index('Speaker:')
            listener_idx = d['input'].find('Listener:')
            if speaker_idx < listener_idx or listener_idx == -1:
                # speaker first
                d['input'] = re.sub('Speaker:', 'Speaker (A):', d['input'])
                d['input'] = re.sub('Listener:', 'Listener (B):', d['input'])

                d['context'] = re.sub('Speaker:', 'Speaker (A):', d['context'])
                d['context'] = re.sub('Listener:', 'Listener (B):', d['context'])
            else:
                # listener first
                d['input'] = re.sub('Listener:', 'Listener (A):', d['input'])
                d['input'] = re.sub('Speaker:', 'Speaker (B):', d['input'])

                d['context'] = re.sub('Listener:', 'Listener (A):', d['context'])
                d['context'] = re.sub('Speaker:', 'Speaker (B):', d['context'])

        if 'soda' in modelname and ('source' not in d or d['source'] == 'None'):
            # show the same speaker/listener tags for all models in display on annotation interface for consistency!!!!
            speaker_idx = d['input'].index('Speaker:')
            listener_idx = d['input'].find('Listener:')
            if speaker_idx < listener_idx or listener_idx == -1:
                # speaker first
                d['context'] = re.sub('Speaker:', 'Speaker (A):', d['context'])
                d['context'] = re.sub('Listener:', 'Listener (B):', d['context'])
            else:
                # listener first
                d['context'] = re.sub('Listener:', 'Listener (A):', d['context'])
                d['context'] = re.sub('Speaker:', 'Speaker (B):', d['context'])

def run(model, data, metrics):
    results = {}
    for metric, kwargs in metrics:
        r = metric(model=model, data=data, **kwargs)
        if isinstance(r, dict):
            results.update({f"{metric.__name__}_{k}": v for k,v in r.items()})
        else:
            results[metric.__name__] = r
    return results


metrics = {
    'functions': [
        (sacrebleu, dict(multianswer=False, prefix='singleanswer_max')),
        (bertscore, dict(multianswer=False, prefix='singleanswer_max')),
        (sentencebert, dict(multianswer=False, prefix='singleanswer_max')),
        (sacrebleu, dict(multianswer=True, prefix='multianswer_linsumassign_top5', max_out=5)),
        (bertscore, dict(multianswer=True, prefix='multianswer_linsumassign_top5', max_out=5)),
        (sentencebert, dict(multianswer=True, prefix='multianswer_linsumassign_top5', max_out=5)),
    ]
}

if __name__ == '__main__':

    gpt_models = [
        GPT_CS_GENERATOR
    ]
    test_datasets = [
        load_gpt_test_set(),
        load_human_test_set(),
    ]

    if sys.argv[1] == 'gpt':
        models = gpt_models
    else:
        raise ValueError(f"First argument should be modeltype: {sys.argv[0]}")

    if sys.argv[2] == 'test':
        datasets = test_datasets
    else:
        raise ValueError(f"Second argument should be dataset type: {sys.argv[1]}")

    output_filepath = sys.argv[3]

    transformation = None
    if sys.argv[4] == 'qmatch':
        def filter_unmatched_questions(data):
            ignore = {
                'What events happened before the situation that Speaker just shared?',
                'What would cause the last thing said to be untrue or unsuccessful?',
                'What does the last thing said cause to happen?',
                'How does the last thing said affect Speaker?',
                'How does the last thing said affect Listener?'
            }
            return {'filtered': [t for t in data if t['question'] not in ignore]}
        transformation = filter_unmatched_questions

    all_outputs = {}
    all_results = {}
    for data_filepath, data in datasets:
        for model_settings in models:
            print('\n' + '#' * 50 + '\n')
            repetition_penalty = model_settings.repetition_penalty
            divbeam_dict = dict(
                num_beams=model_settings.num_beams,
                num_beam_groups=model_settings.num_beam_groups,
                diversity_penalty=model_settings.diversity_penalty,
            )
            modelname = model_settings.modelname
            formatted_data = load_results(
                checkpoint_path=model_settings.modelpath,
                filepath=data_filepath,
                epoch=model_settings.epoch,
                modelname=modelname,
                repetition_penalty=repetition_penalty,
                divbeam_dict=divbeam_dict,
                save_dir='eval/auto/results/'
            )
            if formatted_data is None:
                formatted_data = format_data(
                    data=data,
                    format_string=SELECTED_ARGS[modelname]['format'],
                    convert_q=SELECTED_ARGS[modelname].get('convert_q', False),
                    context_length=SELECTED_ARGS[modelname]['context_length'],
                    to_instr_format=SELECTED_ARGS[modelname]['instruction_format'],
                    newline_delimiter=SELECTED_ARGS[modelname]['newline_delimiter'],
                    prefix=SELECTED_ARGS[modelname]['prefix']
                )
                convert_data_format_per_model(formatted_data, modelname)
            print(model_settings.name)

            model, tokenizer = None, None
            data_updated = False
            if formatted_data is not None and BEAMED_GENERATIONS not in formatted_data[0]:  # beamed generations processing
                data_updated = True
                if 't5' in modelname:
                    model, tokenizer = load_t5(model_settings.modelpath, device='cuda')
                    gen_data_processor = DataCollatorForT5(
                        tokenizer=tokenizer,
                        source_max_len=768,
                        predict_with_generate=True,
                    )
                else:
                    raise ValueError('Only t5 model is supported!')
                get_generations(
                    model=model,
                    tokenizer=tokenizer,
                    data=formatted_data,
                    data_processor=gen_data_processor,
                    batch_size=model_settings.batch_size,
                    num_to_gen=model_settings.num_to_gen,
                    repetition_penalty=repetition_penalty,
                    **divbeam_dict,
                    device='cuda'
                )
                del model

            if data_updated:
                print('saving...')
                save_results(
                    formatted_verified_dev_set=formatted_data,
                    checkpoint_path=model_settings.modelpath,
                    filepath=data_filepath,
                    epoch=model_settings.epoch,
                    modelname=modelname,
                    repetition_penalty=repetition_penalty,
                    divbeam_dict=divbeam_dict,
                    save_dir='eval/auto/results/'
                )

            data_as_key = data_filepath[data_filepath.rindex('/')+1:data_filepath.rindex('.')]
            if transformation is None:
                key = f"{model_settings.name}_{data_as_key}"
                outs = run(
                    metrics=[v for ls in metrics.values() for v in ls],
                    data=formatted_data,
                    model=modelname
                )
                all_results[key] = {
                    k: float(v)
                    if v is not None and not isinstance(v, str)
                    else v
                    for k, v in outs.items()
                }
                print(json.dumps(all_results[key], indent=2))
            else:
                print(f'Transforming data before eval using: {transformation.__name__}')
                transformed_data = transformation(formatted_data)
                for key_suffix, tdata in transformed_data.items():
                    key = f"{model_settings.name}_{key_suffix}_{data_as_key}"
                    outs = run(
                        metrics=[v for ls in metrics.values() for v in ls],
                        data=tdata,
                        model=modelname
                    )
                    all_results[key] = {
                        k: float(v)
                        if v is not None and not isinstance(v, str)
                        else v
                        for k, v in outs.items()
                    }
                    print(json.dumps(all_results[key], indent=2))

            print('saving after metrics...')
            save_results(
                formatted_verified_dev_set=formatted_data,
                checkpoint_path=model_settings.modelpath,
                filepath=data_filepath,
                epoch=model_settings.epoch,
                modelname=modelname,
                repetition_penalty=repetition_penalty,
                divbeam_dict=divbeam_dict,
                save_dir='eval/auto/results/'
            )

    df = pd.DataFrame(all_results).T
    df = df.round(decimals=4)
    print(df)
    if output_filepath != "None":
        df.to_csv(f"eval/auto/results/{output_filepath}")
