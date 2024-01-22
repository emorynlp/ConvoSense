from eval.diversity.gpt_utils import Prompt
import json
import time
import os
import openai
import multiprocessing as mp
import re
import traceback

class AnswerGrouping(Prompt):
    """
    Group answers based on semantic similarity
    """
    template = """-------------------------------------
{}

Question: {}

Answers:
{}
-------------------------------------
Group answers together that are paraphrases, identical in meaning, or very similar in meaning to one another

Your groupings should ensure that:

- Answers with SIMILAR MEANING are in the SAME GROUP, even if their language and words are not exactly the same!
- Answers that are not similar to any other answer are isolated into their own group of size 1.

Please keep in mind: Referring to the same conversational participant in different ways (e.g. by their role in the conversation vs their name, etc.) does NOT change the underlying meaning, so such answers should still be grouped together.

Group 1:
("""

def parse_groupings(generated):
    try:
        generated = 'Group 1:\n(' + generated
        generated_lines = [x.strip() for x in generated.split('\n') if x.strip() != '']
        groups = {}
        curr_group = None
        curr_group_items = None
        for line in generated_lines:
            if line.startswith('Group'):
                if curr_group is not None:
                    groups[curr_group] = curr_group_items
                curr_group_items = []
                curr_group = re.findall('(Group \d+)', line)[0]
            else:
                answer_idx = re.findall('^\(?(\d+)\)', line)
                if answer_idx:
                    answer_idx = answer_idx[0]
                    curr_group_items.append((answer_idx, line[line.index(')') + 1:].strip()))
                else:
                    print('WEIRD!!!!')
                    print(line)
        if curr_group_items:
            groups[curr_group] = curr_group_items
        return groups
    except Exception as e:
        print('ERROR!!!!')
        print(generated)
        traceback.print_exc()
        return {}

def gen(tasks, k, file_prefix):
    openai.api_key = os.getenv("OPENAI")
    generations = []
    total_tokens = 0
    ts = time.time()
    for j, (context, question, answers, item) in enumerate(tasks):
        if (j+1) % 5 == 0:
            print(f'Process {k} is starting task {j + 1} of {len(tasks)}', flush=True)
        generator = AnswerGrouping(context, question, answers, parse=parse_groupings, temperature=0.0)
        s = time.perf_counter()
        generated, output, used_tokens = generator.generate(modelname='gpt-4-0613')
        dur = time.perf_counter() - s
        generations.append({
            **item,
            'elapsed': f"{int(dur)}",
            'prompt': generator.prompt,
            'generated': generated,
            'output': output,
            'full': generator.prompt.split('\n') + generated.split('\n'),
            'used_tokens': used_tokens
        })
        total_tokens += used_tokens
    print(f'Process {k} completed {j + 1}/{len(tasks)} tasks')
    print(f'\t{total_tokens:,} tokens used in {(time.time() - ts) / 60:.1f} min', flush=True)
    json.dump(generations, open(f'{file_prefix}{k}th_proc.json', 'w'), indent=2)

def gen_multiprocess(tasks, procs, file_prefix):
    tasks = [tasks[i::procs] for i in range(procs)]
    with mp.Pool(procs) as pool:
        pool.starmap(
            gen,
            [(tasks[i], i, file_prefix) for i in range(procs)]
        )

def gpt_diversity_eval_of_models():
    data = json.load(open('eval/diversity/inputs/gptaseval_0_100_best_human_mono_bs-best_gpt_mono_bs-best_human_only_poly_diversity.json'))
    # data = json.load(open('eval/diversity/inputs/gptaseval_0_100_best_human_mono_dbs-best_gpt_mono_dbs-best_gpt_poly_diversity.json')) # uncomment this to get GPT diversity eval results on these 3 models

    tasks_to_do = []
    model_counts = {}
    model_items = {}
    for item in data:
        source = 'gpt' if 'None' in item['source'] else 'human'
        modelname = item['model_settings-modelname']
        question = item['question']
        if question not in model_counts:
            model_counts[question] = {}
        if source not in model_counts[question]:
            model_counts[question][source] = {}
        if modelname not in model_counts[question][source]:
            model_counts[question][source][modelname] = 0
        if model_counts[question][source][modelname] < 50:
            answers_ls = [v for k,v in item.items() if k.startswith('Answer') and '_score' not in k and v is not None]
            tasks_to_do.append(
                (item['context'], item['question'], '\n'.join(f"({i+1}) {a}" for i, a in enumerate(answers_ls)), item)
            )
            model_counts[question][source][modelname] += 1
            model_items.setdefault(question, {}).setdefault(source, {}).setdefault(modelname, []).append(f"{item['context']}_{question}")
    for question, sources in model_items.items():
        for source, outputs in sources.items():
            for matches in list(zip(*list(outputs.values()))):
                for other in matches[1:]:
                    assert matches[0] == other

    dir = 'eval/diversity/annotations/new_run/'
    if not os.path.exists(dir):
        print('making dir: ', dir)
        os.mkdir(dir)

    print('starting gen...')
    gen_multiprocess(
        tasks=tasks_to_do,
        procs=5,
        file_prefix=dir
    )

if __name__ == '__main__':
    gpt_diversity_eval_of_models()
