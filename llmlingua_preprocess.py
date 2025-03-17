from datasets import load_dataset
from llmlingua import PromptCompressor
import pandas as pd
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def compress_llm(prompt, rate=0.33, force_tokens = ['\n', '?']):
    llm_lingua = PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True, # Whether to use llmlingua-2
        )
    return llm_lingua.compress_prompt(prompt, rate=rate, force_tokens = force_tokens)['compressed_prompt']

if __name__ == "__main__":
    tqdm.pandas()
    # Load the dataset
    dataset_names = ["multifieldqa_en", "hotpotqa", "triviaqa", "narrativeqa"]
    for dataset_name in dataset_names:
        print("Processing dataset:", dataset_name)
        dataset = load_dataset('THUDM/LongBench', dataset_name, split='test')
        df = pd.DataFrame(dataset)
        compression_rates = [0.1,0.4,0.6,0.9]
        for i in tqdm(compression_rates, desc=f"Compressing {dataset_name}"):
            df['context_' + str(i)] = df['context'].progress_apply(lambda x: compress_llm(x, rate=i))
        df = df.sort_index(axis=1)
        df.to_csv("llmlingua_"+dataset_name+'.csv')

