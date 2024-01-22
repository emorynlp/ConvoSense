# ConvoSense

Paper: [**ConvoSense:  Overcoming Monotonous Commonsense Inferences for Conversational AI**](paper.pdf) (TACL 2024)

While there have been several attempts to create datasets that facilitate commonsense inferences in dialogue contexts, existing datasets tend to lack in-depth details, restate information already present in the conversation, and often fail to capture the multifaceted nature of commonsense reasoning.

In response to these limitations, we compile a new synthetic dataset for commonsense reasoning in dialogue contexts using GPT, ConvoSense, that boasts greater contextual novelty, offers a higher volume of inferences per example, and substantially enriches the detail conveyed by the inferences.

Our dataset contains over 500,000 inferences across 12,000 dialogues with 10 popular inference types, which empowers the training of generative commonsense models for dialogue that are superior in producing plausible inferences with high novelty when compared to models trained on the previous datasets. 

We release our dataset ConvoSense and our best-performing dialogue commonsense generative model ConvoSenseM* in this repository.

## Dependencies

* Python >=3.9
* `requirements.txt`
    * Usage: pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
    * IMPORTANT: Make sure to update the `torch` installation to be compatible with the GPU on your machine. 

## Data

NOTE: Due to their size, `tar.gz` files are provided of the included data in `data/`. Once you clone the repository, you must extract the files (e.g. on Linux: `tar -xzvf convosense.tar.gz` to get `convosense/` data).

The train, dev, and test splits of the ConvoSense dataset are located at `data/convosense/{train|dev|test}.jsonl`.

Each datapoint is composed of a single output inference for the specified dialogue context and commonsense type. 

Since ConvoSense was generated to include multiple inferences per context and type, there are multiple datapoints that share the same dialogue context and commonsense type ("input"), yet have a different output inference ("output").

*Datapoint Format:*

```json
{
    "id": "soda_7255_batman_safia_spiderman_tickets", # dialogue id
    "type": "cause", # commonsense type
    "question": "What could have caused the last thing said to happen?", # commonsense type as question
    "context": "Listener: Hey Beatriz, do you want to go see a movie with me?\nSpeaker: Sure, that sounds like fun. What movie do you want to see?\nListener: I don't know, there are a lot of good ones out right now. Do you have any suggestions?\nSpeaker: Well, I've been wanting to see the new Batman movie.", # dialogue context
    "all_answers": [
        "the speaker's interest in superhero films.", 
        "the speaker seeing a trailer for the new batman movie.", 
        "the speaker's desire to see the latest blockbuster film.", 
        "the speaker being excited about the cast and director of the new batman movie.", 
        "the speaker hearing positive reviews about the new batman movie from friends or online."
    ], # all GPT-generated commonsense inferences of specified type for the dialogue context
    "input": "Listener: Hey Beatriz, do you want to go see a movie with me?\nSpeaker: Sure, that sounds like fun. What movie do you want to see?\nListener: I don't know, there are a lot of good ones out right now. Do you have any suggestions?\nSpeaker: Well, I've been wanting to see the new Batman movie.\n\n[Question] What could have caused the last thing said to happen?\n[Answer]", # datapoint converted to model input format 
    "output": "the speaker's interest in superhero films.",  # datapoint converted to model output format 
}
```

Parsed versions of the data are also available at `data/{train/dev/test}_onlydialogues.json`. These parsed versions were compiled by executing `data/parse_format.py` to convert the original data into an easier-to-use format.

These parsed versions are compatible with the provided data loading functions in `data/load.py` which loads the data as a `Dialogues` object from `data/dialogues_struct.py`:

```python
import data.load as data_loader
data = data_loader.load_data(file='test_onlydialogues.json')
```

This `Dialogues` object is used in the commonsense inference generation script at `gen/cs_generation.py` for predicting inferences using the trained model.

HumanGen dataset files are also available at `data/humangen/{train|dev|test}.jsonl`.

## Trained Model

NOTE: Due to its large size, we are still working on directly providing the model checkpoint. For now, if you would like access to the model, please send an email to "sfillwo@emory.edu". 

Our best-performing ConvoSense-trained model (ConvoSenseM*) is located at `models/convosense_m_star`.

The script at `gen/cs_generation.py` is used for predicting inferences using the trained model.

## Evaluation Results

### GPT vs Human Inferences Human Evaluation (Section 3.3)

`eval/gpt_vs_human/` contains the code and data for running the analyses to compare GPT and Human inferences on the datapoints from existing human-written datasets. The script for computing the results seen in paper Section 3.3 is `eval/gpt_vs_human/compute.py`. The annotations performed by Surgers that these analyses are based on can be found in `eval/gpt_vs_human/annotations/`.

### ConvoSense & Models Human Evaluation (Sections 4.2 and 6.3)

`eval/human/` contains the code and data for running the analyses on the human evaluations of the models (paper Section 6.3) and ConvoSense data (paper Section 4.2). The script for computing the results is `eval/human/compute.py`. The annotations performed by the human expert that these analyses are based on can be found in `eval/human/{convosense|models}/annotations/`.

### Models Automatic Evaluation (Section 6.1)

To calculate automatic metrics on HumanGen and ConvoSense test sets for the best ConvoSense-trained model (ConvoSenseM*):

`python eval/auto/compute.py gpt test gpt_test_metrics_savedscores_bleu.csv qmatch`

### Models Automatic Diversity Evaluation (Section 6.2)

`eval/diversity/` contains the code and data for running the analyses on the GPT4 diversity evaluations of the models (paper Section 6.2). The script for computing the results is `eval/diversity/compute.py`. The annotations outputted by GPT4 that these analyses are based on can be found in `eval/diversity/annotations/`. The script that was used to get the GPT4 diversity annotations is `eval/diversity/gpt_as_diversity_evaluator.py`.

You need to put your OpenAI api key into an environment variable called `OPENAI`.

## Interface for GPT Collection

The code used to generated ConvoSense using ChatGPT3.5 is located in `data/soda/`.

`data/soda/convosense_generation.py` contains the script that was executed for the full data collection process.

NOTE: This is intended to be documentation on the collection process; there is no need to rerun this code since the ConvoSense dataset is provided at `data/convosense`!

## Reproduce Model Training

The code used for training is located at `train/`. The training script `run_seq2seq.py` was adapted from the [CICERO repository](https://github.com/declare-lab/CICERO). Training was performed using [SLURM](https://slurm.schedmd.com/quickstart.html) for GPU management, with the execution shell script `train.sh`.








