import os
import pickle
from tqdm import tqdm
from dataclasses import dataclass
import transformers
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
)
from eval.auto.global_vars import BEAMED_GENERATIONS, SELECTED_ARGS, BEAMED_GENERATIONS_SCORES

def get_output_filepaths(checkpoint_path, epoch, modelname, filepath, repetition_penalty, divbeam_dict):
    if filepath is None:
        return None
    output_dir = checkpoint_path.replace('/', '-') if epoch is not None else modelname
    output_dataname = filepath.replace('/', '-').replace('.json', '') + (f"_rp{repetition_penalty}" if repetition_penalty != 1.0 else '') + (''.join([f"_{k}{v}" for k,v in divbeam_dict.items() if k not in {'batch_size'}]) if divbeam_dict else '')
    print(output_dir)
    print(output_dataname)
    return output_dir, output_dataname

def load_results(checkpoint_path, filepath, epoch, modelname, repetition_penalty, divbeam_dict, save_dir='evaluation/model_outputs'):
    if filepath is None:
        return None
    output_dir, output_dataname = get_output_filepaths(checkpoint_path, epoch, modelname, filepath, repetition_penalty, divbeam_dict)
    if os.path.exists(f"{save_dir}/{output_dir}/{output_dataname}.pickle"):  # load saved data
        with open(f"{save_dir}/{output_dir}/{output_dataname}.pickle", "rb") as file:
            formatted_verified_dev_set = pickle.load(file)
        return formatted_verified_dev_set
    return None

def save_results(formatted_verified_dev_set, checkpoint_path, epoch, modelname, filepath, repetition_penalty, divbeam_dict, save_dir='evaluation/model_outputs'):
    if filepath is None:
        return
    output_dir, output_dataname = get_output_filepaths(checkpoint_path, epoch, modelname, filepath, repetition_penalty, divbeam_dict)
    if not os.path.exists(f"{save_dir}/{output_dir}"):
        os.mkdir(f"{save_dir}/{output_dir}")
    with open(f"{save_dir}/{output_dir}/{output_dataname}.pickle", "wb") as file:
        pickle.dump(formatted_verified_dev_set, file, protocol=pickle.HIGHEST_PROTOCOL)


###########################################
        
to_target_questions = {
    'What might happen after what Speaker just said?': 'What subsequent event happens or could happen following the target?',
    'What could have caused the last thing said to happen?': 'What is or could be the cause of target?',
    'How does Listener feel because of what Speaker just said?': 'What is the possible emotional reaction of the listener in response to target?',
    'How does Listener (or others) feel because of what Speaker just said?': 'What is the possible emotional reaction of the listener in response to target?',
    'What is an emotion or human drive that motivates Speaker based on what they just said?': 'What is or could be the motivation of target?',

    'What does Speaker want to do next?': 'What does the speaker of target want to do next?',
    'What events happened before the situation that Speaker just shared?': 'What events happened before the situation shared in the target?',
    'How is Speaker feeling after what they just said?': 'How is the speaker of the target feeling?',
    'What is a likely characteristic of Speaker based on what they just said?': 'What is a likely characteristic of the speaker of target?',
    'How does the last thing said affect Speaker?': 'How does the target affect the speaker?',
    'What does the last thing said cause to happen?': 'What does the target cause to happen?',
    'What would cause the last thing said to be untrue or unsuccessful?': 'What would cause the target to be untrue or unsuccessful?',
    'What will Listener (or others) want to do next based on what Speaker just said?': 'What will the listener want to do next due to the target?',
    'What will Listener want to do next based on what Speaker just said?': 'What will the listener want to do next due to the target?',
    'What prerequisites are required for the last thing said to occur?': 'What prerequisites are required for the target to occur?',
    'What is a breakdown of the last thing said into a series of required subevents?': 'What is a breakdown of the target into a series of required subevents?',
    'How does the last thing said affect Listener (or others)?': 'How does the target affect the listener?',
    'How does the last thing said affect Listener?': 'How does the target affect the listener?',
}

def context_to_input(item, format_string, convert_q, context_length):
    context_str = item['context']
    context_lines = context_str.split('\n')
    if context_length is not None:
        context_lines = context_lines[-context_length:]
    target = context_lines[-1].replace('Speaker: ', '')
    context = ' <utt> '.join(context_lines)
    if convert_q:
        q = to_target_questions[item['question']]
    else:
        q = item['question']
    input = format_string.format(question=q, target=target, context=context)
    return input

def format_data(data, format_string, convert_q, context_length, to_instr_format, newline_delimiter, prefix=None, disable=False):
    if data is None:
        return None
    formatted_data = []
    for item in tqdm(data, desc='Formatting data', disable=disable):
        input = context_to_input(item, format_string, convert_q, context_length)
        if newline_delimiter:
            input = input.replace(' <utt> ', '\n')
        if prefix:
            input = prefix + input
        formatted_item = {**item, "input": input}
        formatted_data.append(formatted_item)
    return formatted_data

###########################################

def split_ordered_answers(str, remove_num=True):
    splits = [a[a.index(')') + 2:] if remove_num else a for a in str.split('; (')]
    if splits[-1].endswith(';'):
        splits[-1] = splits[-1][:-1]
    elif len(splits) == 1: # only one answer found (without terminal `;`) but keep it rather than removing it
        pass
    else: # if last element in splits does NOT end with ';' max_new_tokens was encountered before generation was finished!
        splits = splits[:-1]
    return splits

def extract_multianswers(example, modelname, only_one_beam=True, remove_num=True):
    all_answers = example[BEAMED_GENERATIONS]
    if only_one_beam:
        if '(1) ' in all_answers[0]:
            all_answers = split_ordered_answers(all_answers[0], remove_num=remove_num)
        else:
            all_answers = [all_answers[0]]
    else:
        new_all_answers = None
        for gen in all_answers:
            if '(1) ' in gen:
                new_all_answers = [] if new_all_answers is None else new_all_answers
                new_all_answers.extend(split_ordered_answers(gen, remove_num=remove_num))
        all_answers = new_all_answers if new_all_answers is not None else all_answers
    return all_answers

###########################################

def load_t5(checkpoint, device='cpu', torch_dtype=None):
    if torch_dtype is None:
        model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)
    else:
        model = T5ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch_dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model.config.n_positions = 768
    return model, tokenizer

@dataclass
class DataCollatorForT5(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    predict_with_generate: bool

    def __call__(self, instances):
        inputs = [d['input'] for d in instances]
        model_inputs = self.tokenizer(inputs, max_length=self.source_max_len, padding=True, truncation=True, return_tensors='pt')
        if not self.predict_with_generate:
            model_inputs['labels'] = self.get_label_encoding([d['output'] for d in instances])
        return model_inputs

    def get_label_encoding(self, labels):
        label_encoding = self.tokenizer(labels, return_tensors='pt', padding=True).input_ids
        label_encoding[label_encoding == 0] = -100
        return label_encoding

def get_generations(model, tokenizer, data, data_processor, batch_size, num_to_gen, repetition_penalty=1.0,
                    num_beams=5, num_beam_groups=1, diversity_penalty=None, device='cuda', disable=False,
                    return_dict_in_generate=False, output_scores=False):
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    for k in tqdm(range(0, len(data), batch_size), desc='Getting generations', disable=disable):
        items_ = data[k: min(k + batch_size, len(data))]
        items = data_processor([{'input': item['input'], 'output': ''} for item in items_])
        input_ids = items["input_ids"]
        attention_mask = items["attention_mask"]
        beam_outputs = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_new_tokens=400,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            early_stopping=True,
            num_return_sequences=num_to_gen,
            repetition_penalty=repetition_penalty,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
        )
        if return_dict_in_generate and output_scores:
            for idx, chunk in enumerate(zip(
                    chunks(beam_outputs.sequences, num_to_gen),
                    chunks(beam_outputs.sequences_scores, num_to_gen),
                )
            ):
                if isinstance(model, T5ForConditionalGeneration):
                    beamed_generations = tokenizer.batch_decode(chunk[0], skip_special_tokens=True)
                    sequence_scores = chunk[1].tolist()
                    items_[idx][BEAMED_GENERATIONS_SCORES] = sequence_scores
                else:
                    generated_ids = _remove_leading_elements(input_ids[idx], chunk[0])
                    beamed_generations = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                items_[idx][BEAMED_GENERATIONS] = beamed_generations
        else:
            for idx, chunk in enumerate(chunks(beam_outputs, num_to_gen)):
                if isinstance(model, T5ForConditionalGeneration):
                    beamed_generations = tokenizer.batch_decode(chunk, skip_special_tokens=True)
                else:
                    generated_ids = _remove_leading_elements(input_ids[idx], chunk)
                    beamed_generations = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                items_[idx][BEAMED_GENERATIONS] = beamed_generations