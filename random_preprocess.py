from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import warnings
import random

# Suppress warnings
warnings.filterwarnings("ignore")

def random_prune(prompt, rate=0.33):
    # Randomly prune the prompt by the given rate
    tokens = prompt.split()
    num_tokens_to_prune = int(len(tokens) * rate)
    pruned_tokens = random.sample(tokens, num_tokens_to_prune)
    pruned_prompt = ' '.join(pruned_tokens)
    return pruned_prompt
    

if __name__ == "__main__":
    tqdm.pandas()
    # Load the dataset
    dataset_names = ["multifieldqa_en", "hotpotqa", "triviaqa", "narrativeqa"]
    for dataset_name in dataset_names:
        print("Processing dataset:", dataset_name)
        dataset = load_dataset('THUDM/LongBench', dataset_name, split='test')
        df = pd.DataFrame(dataset)
        compression_rates = [0.1,0.25,0.4,0.5,0.6,0.75,0.9]
        for i in tqdm(compression_rates, desc=f"Compressing {dataset_name}"):
            df['context_' + str(i)] = df['context'].progress_apply(lambda x: random_prune(x, rate=i))
        df = df.sort_index(axis=1)
        df.to_csv("random_prune_"+dataset_name+'.csv')

