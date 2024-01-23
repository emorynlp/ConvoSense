from tqdm import tqdm

##########################################################################################################
##
##    PARAMS
##
##########################################################################################################


BEAMED_GENERATIONS = 'beamed_generations'
BEAMED_GENERATIONS_SCORES = 'beamed_generations_scores'


COMMONSENSE_LS = [
    'What could have caused the last thing said to happen?',  # cause
    'What prerequisites are required for the last thing said to occur?',  # prerequisities
    'What is an emotion or human drive that motivates Speaker based on what they just said?',  # motivation
    'What might happen after what Speaker just said?',  # subsequent
    'What does Speaker want to do next?',  # desire
    'What will Listener want to do next based on what Speaker just said?',  # desire_o
    'How is Speaker feeling after what they just said?',  # react
    'How does Listener feel because of what Speaker just said?',  # react_o
    'What is a likely characteristic of Speaker based on what they just said?',  # attribute
    'What is a breakdown of the last thing said into a series of required subevents?',  # constituents
]

##########################################################################################################
##
##    DATA FORMAT
##
##########################################################################################################


def context_to_input(item, format_string, context_length):
    context_str = item['context']
    context_lines = context_str.split('\n')
    if context_length is not None:
        context_lines = context_lines[-context_length:]
    target = context_lines[-1].replace('Speaker: ', '')
    context = '\n'.join(context_lines)
    q = item['question']
    input = format_string.format(question=q, target=target, context=context)
    return input


def format_data(data, format_string, context_length, prefix=None, disable_tqdm=False):
    if data is None:
        return None
    formatted_data = []
    for item in tqdm(data, desc='Formatting data', disable=disable_tqdm):
        input = context_to_input(item, format_string, context_length)
        if prefix:
            input = prefix + input
        formatted_item = {**item, "input": input}
        formatted_data.append(formatted_item)
    return formatted_data

def add_speaker_to_lines(lines: list[str]) -> str:
    elements = list(
        reversed(
            [
                f"{'Speaker' if i % 2 == 0 else 'Listener'}: {u}"
                for i, u in enumerate(reversed(lines))
            ]
        )
    )
    dia = '\n'.join(elements)
    return dia

def chunks(lst, n):
    """Return successive n-sized chunks from lst."""
    outs = []
    for i in range(0, len(lst), n):
        outs.append(lst[i:i + n])
    return outs