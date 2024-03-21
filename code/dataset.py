from datasets import load_dataset

import json

def get_dataset(datasets):
    if 'laion' in datasets:
        dataset = load_dataset(datasets)['train']
        prompt_key = 'TEXT'
    elif 'coco' in datasets:
        with open('fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)
            dataset = dataset['annotations']
            prompt_key = 'caption'
    else:
        dataset = load_dataset(datasets)['test']
        prompt_key = 'Prompt'

    return dataset, prompt_key