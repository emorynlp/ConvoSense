import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("sefinch/ConvoSenseGenerator")
model = T5ForConditionalGeneration.from_pretrained("sefinch/ConvoSenseGenerator").to(device)

# ConvoSenseGenerator covers these commonsense types, using the provided questions
commonsense_questions = {
    "cause": 'What could have caused the last thing said to happen?', 
    "prerequisities": 'What prerequisites are required for the last thing said to occur?', 
    "motivation": 'What is an emotion or human drive that motivates Speaker based on what they just said?', 
    "subsequent": 'What might happen after what Speaker just said?', 
    "desire": 'What does Speaker want to do next?',
    "desire_o": 'What will Listener want to do next based on what Speaker just said?',
    "react": 'How is Speaker feeling after what they just said?',
    "react_o": 'How does Listener feel because of what Speaker just said?',
    "attribute": 'What is a likely characteristic of Speaker based on what they just said?',
    "constituents": 'What is a breakdown of the last thing said into a series of required subevents?' 
}

def format_input(conversation_history, commonsense_type):

    # prefix last turn with Speaker, and alternately prefix each previous turn with either Listener or Speaker
    prefixed_turns = list(
        reversed(
            [
                f"{'Speaker' if i % 2 == 0 else 'Listener'}: {u}"
                for i, u in enumerate(reversed(conversation_history))
            ]
        )
    )

    # model expects a maximum of 7 total conversation turns to be given
    truncated_turns = prefixed_turns[-7:]

    # conversation representation separates the turns with newlines
    conversation_string = '\n'.join(truncated_turns)

    # format the full input including the commonsense question
    input_text = f"provide a reasonable answer to the question based on the dialogue:\n{conversation_string}\n\n[Question] {commonsense_questions[commonsense_type]}\n[Answer]"

    return input_text

def generate(conversation_history, commonsense_type):
    # convert the input into the expected format to run the model
    input_text = format_input(conversation_history, commonsense_type) 

    # tokenize the input_text
    inputs = tokenizer([input_text], return_tensors="pt").to(device)

    # get multiple model generations using the best-performing generation configuration (based on experiments detailed in paper)
    outputs = model.generate(
        inputs["input_ids"],
        repetition_penalty=1.0,
        num_beams=10,
        num_beam_groups=10,
        diversity_penalty=0.5,
        num_return_sequences=5,
        max_new_tokens=400
    )

    # decode the generated inferences
    inferences = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return inferences

conversation = [
    "Hey, I'm trying to convince my parents to get a dog, but they say it's too much work.",
    "Well, you could offer to do everything for taking care of it. Have you tried that?",
    "But I don't want to have to take the dog out for walks when it is the winter!"
]

inferences = generate(conversation, "cause")
print('\n'.join(inferences))