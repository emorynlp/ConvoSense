import os
from models.model_settings import GenerationConfig, DataConfig
from models.t5 import CommonsenseGenerator

GPT_CS_GENERATOR = CommonsenseGenerator(
    name="ConvoSenseGenerator",
    modelpath=f'sefinch/ConvoSenseGenerator',
    device='cuda',
    use_commonsense=False,
    generation_config=GenerationConfig(
        repetition_penalty=1.0,
        num_beams=10,
        num_beam_groups=10,
        diversity_penalty=0.5,
        temperature=None,
        top_p=None
    ),
    data_config=DataConfig(
        context_length=7,
        format="{context}\n\n[Question] {question}\n[Answer]",
        prefix="provide a reasonable answer to the question based on the dialogue:\n"
    ),
    batch_size=4,
    num_to_gen=10
)