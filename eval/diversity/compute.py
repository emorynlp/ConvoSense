import os, json, re
from copy import deepcopy
import string
import pandas as pd

def parse_groupings(generated):
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
            answer_idx = re.findall('^\((\d+)\)', line)
            if answer_idx:
                answer_idx = answer_idx[0]
                curr_group_items.append((answer_idx, line[line.index(')')+1:].strip()))
    if curr_group_items:
        groups[curr_group] = curr_group_items
    return groups

def load_gpt_diversity_outputs(dir, files=None, add_answer_keys=False):
    """
    Load output file from GPT diversity task
    """
    if files is None:
        files = os.listdir(dir)
        for file in list(files):
            if os.path.isdir(f"{dir}/{file}"):
                nested_files = [f"{file}/{nf}" for nf in os.listdir(f"{dir}/{file}")]
                files.extend(nested_files)
                files.remove(file)
    elif isinstance(files, str):
        files = [files]
    gpt_diversity_results = {}
    total = 0
    odd = 0
    missing = 0
    too_many = 0
    for file in files:
        data = json.load(open(f"{dir}/{file}"))
        for item_dict in data:
            if add_answer_keys:
                for i, answer in enumerate(item_dict["all_answers"]):
                    item_dict[f"Answer{i+1}"] = answer
                    item_dict[f"Answer{i+1}_score"] = 100 - i
                item_dict['model_settings-name'] = 'gpt'
                item_dict['source'] = 'convosense'
            parsed_groups = {}
            total += 1
            for k, ls in item_dict['output'].items():
                parsed_groups[k] = []
                for v in ls:
                    if f"Answer{v[0]}" in item_dict:  # verify that outputted answer exists
                        if item_dict[f"Answer{v[0]}"].strip().lower() == v[1].strip().lower():  # verify that outputted answer matches text of inputted answers
                            answer_info = (item_dict[f"Answer{v[0]}"], f"Answer{v[0]}", item_dict[f"Answer{v[0]}_score"])
                            parsed_groups[k].append(answer_info)
                        elif v[1].translate(str.maketrans('', '', string.punctuation)).strip().lower() == item_dict[f"Answer{v[0]}"].translate(str.maketrans('', '', string.punctuation)).strip().lower(): # sometimes, gpt outputs different punctuation than inputted answers
                            answer_info = (item_dict[f"Answer{v[0]}"], f"Answer{v[0]}", item_dict[f"Answer{v[0]}_score"])
                            parsed_groups[k].append(answer_info)
                        else:
                            odd += 1
                    else:
                        odd += 1
            item_dict['parsed_groups'] = {k:v for k,v in parsed_groups.items() if v}
            # remove duplicate groups
            duplicates = set()
            for g, ls in item_dict['parsed_groups'].items():
                for other, other_ls in item_dict['parsed_groups'].items():
                    if g != other:
                        if ls == other_ls:
                            duplicates.add(tuple(sorted((g, other))))
            if duplicates:
                for g1, g2 in duplicates:
                    del item_dict['parsed_groups'][g2]
            # remove isolated group if answer exists in other group? Only 4 examples affected so don't do this for now
            # if gpt is missing an answer, then ignore it for that example (only 1% of results)
            all_answers = {k for k,v in item_dict.items() if k.startswith('Answer') and '_score' not in k and v is not None}
            covered_answers = [v[1] for k, ls in item_dict['parsed_groups'].items() for v in ls]
            covered_answers_set = set(covered_answers)
            if len(covered_answers) != len(covered_answers_set):  # verify each answer only used in one group
                too_many += 1
            uncovered_answers = all_answers - covered_answers_set
            if uncovered_answers:
                missing += 1
            item_dict['worker_id'] = 'gpt4'
            id = len(gpt_diversity_results)
            # get top-5 groups too
            sorted_answers = sorted(
                [v for group, ls in item_dict['parsed_groups'].items() for v in ls],
                key=lambda x: x[2],
                reverse=True
            )
            top_5_answers = sorted_answers[:5]
            top_5_parsed_groups = deepcopy(item_dict['parsed_groups'])
            for group, answer_ls in item_dict['parsed_groups'].items():
                for answer in answer_ls:
                    if answer not in top_5_answers:
                        top_5_parsed_groups[group].remove(answer)
            item_dict['parsed_groups_top5'] = {k:v for k,v in top_5_parsed_groups.items() if v}
            gpt_diversity_results[f"output{id}"] = [item_dict]

    return gpt_diversity_results



abr_to_q = {
    'Ant': ("What events happened before the situation that Speaker just shared?", "Before this, ..."),
    'Eff': ("What does the last thing said cause to happen?", "This causes..."),
    'Eff_s': ("How does the last thing said affect Speaker?", "This causes Speaker to..."),
    'Eff_o': ("How does the last thing said affect Listener?", "This causes Listener to..."),
    'Sub': ("What might happen after what Speaker just said?", "After this, ..."),
    'Pre': ("What prerequisites are required for the last thing said to occur?", "For this to happen, it must be true that..."),
    'Cau': ("What could have caused the last thing said to happen?", "This was caused by..."),
    'Att': ("What is a likely characteristic of Speaker based on what they just said?", "Speaker is..."),
    'Mot': ("What is an emotion or human drive that motivates Speaker based on what they just said?", "Speaker is motivated..."),
    'Rea': ("How is Speaker feeling after what they just said?", "Speaker feels..."),
    'Rea_o': ("How does Listener feel because of what Speaker just said?", "Listener feels..."),
    'Des': ("What does Speaker want to do next?", "As a result, Speaker wants..."),
    'Des_o': ("What will Listener want to do next based on what Speaker just said?", "As a result, Listener wants..."),
    'Con': ("What is a breakdown of the last thing said into a series of required subevents?", "This involves..."),
    'Obs': ("What would cause the last thing said to be untrue or unsuccessful?", "This is untrue or unsuccessful if...")
}
q_to_abr = {v[0]: k for k,v in abr_to_q.items()}

if __name__ == '__main__':

    gpt4_diversity_results_for_models = load_gpt_diversity_outputs(
        dir='eval/diversity/annotations/'
    )

    num_of_unique = {}
    ratio_of_unique = {}
    all_overlap_metrics = {}
    header = ['model', 'question', 'source', 'sourcetype', 'num_of_unique', 'ratio_of_unique', 'count']
    records = []

    for _, items in gpt4_diversity_results_for_models.items():
        item = items[0]
        modelname = item['model_settings-name']
        groups = item['parsed_groups']
        num_groups = len(groups)
        all_answers = [(v, item[f"{k}_score"]) for k,v in item.items() if k.startswith('Answer') and '_score' not in k and v is not None]
        if 'poly' in modelname or ('gpt' in modelname and 'mono' not in modelname):
            # poly models
            num_total_answers = len([v for k, ls in groups.items() for v in ls])
            records.append(
                (
                    modelname,
                    q_to_abr[item['question']],
                    item['source'][:5],
                    'gpt' if str(item['source']) == 'None' else 'human',
                    num_groups,
                    num_groups / num_total_answers,
                    len(all_answers)
                )
            )
        else:
            # model monos - just look at top 5
            modelname = f"{modelname}-top5"
            groups = item['parsed_groups_top5']
            num_groups = len(groups)
            num_total_answers = len([v for k, ls in groups.items() for v in ls])
            sorted_answers = sorted(
                all_answers,
                key=lambda x: x[1],
                reverse=True
            )
            top_5_answers = sorted_answers[:5]
            records.append(
                (
                    modelname,
                    q_to_abr[item['question']],
                    item['source'][:5],
                    'gpt' if str(item['source']) == 'None' else 'human',
                    num_groups,
                    num_groups / num_total_answers,
                    len(top_5_answers)
                )
            )

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    diversity_df = pd.DataFrame.from_records(records, columns=header)
    all_data_stats = diversity_df.groupby('model').mean(numeric_only=True)
    print(all_data_stats)
    print()

    import scipy.stats as stats

    all_model_results = diversity_df.groupby('model')
    gpt_mono_dbs = all_model_results.get_group('best_gpt_mono_dbs-top5')
    gpt_mono_bs = all_model_results.get_group('best_gpt_mono_bs-top5')
    gpt_poly = all_model_results.get_group('best_gpt_poly')

    t_stat, p_value = stats.ttest_ind(gpt_mono_dbs['num_of_unique'], gpt_mono_bs['num_of_unique'])
    print(f"\t\tgpt_mono_dbs vs gpt_mono_bs: {t_stat:.3f} {p_value:.3f}")

    t_stat, p_value = stats.ttest_ind(gpt_mono_dbs['num_of_unique'], gpt_poly['num_of_unique'])
    print(f"\t\tgpt_mono_dbs vs gpt_poly: {t_stat:.3f} {p_value:.3f}")

    human_mono_dbs = all_model_results.get_group('best_human_mono_dbs-top5')
    human_mono_bs = all_model_results.get_group('best_human_mono_bs-top5')
    human_poly = all_model_results.get_group('best_human_only_poly')

    t_stat, p_value = stats.ttest_ind(human_mono_dbs['num_of_unique'], human_mono_bs['num_of_unique'])
    print(f"\t\thuman_mono_dbs vs human_mono_bs: {t_stat:.3f} {p_value:.3f}")

    t_stat, p_value = stats.ttest_ind(human_mono_dbs['num_of_unique'], human_poly['num_of_unique'])
    print(f"\t\thuman_mono_dbs vs human_poly: {t_stat:.3f} {p_value:.3f}")

    t_stat, p_value = stats.ttest_ind(gpt_mono_dbs['num_of_unique'], human_mono_dbs['num_of_unique'])
    print(f"\t\tgpt_mono_dbs vs human_mono_dbs: {t_stat:.3f} {p_value:.3f}")
