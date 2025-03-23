from datasets import load_dataset
from llmlingua import PromptCompressor
import pandas as pd
from tqdm import tqdm
import time
import torch
import pandas as pd
import numpy as np
import threading
from tqdm import tqdm
import logging

class GPUMemoryMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.measurements = []
        self.running = False

    def start(self):
        self.running = True
        self.measurements = []

        # Monitor in a loop
        while self.running and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            self.measurements.append(memory_allocated)
            time.sleep(self.interval)

    def stop(self):
        self.running = False

    def get_stats(self):
        if not self.measurements:
            return {"avg": 0, "max": 0, "min": 0}

        return {
            "avg": np.mean(self.measurements),
            "max": np.max(self.measurements),
            "min": np.min(self.measurements),
            "measurements": self.measurements
        }

def compress_llm(prompt, rate=0.33, force_tokens = ['\n', '?']):
    llm_lingua = PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True, # Whether to use llmlingua-2
        )
    return llm_lingua.compress_prompt(prompt, rate=rate, force_tokens = force_tokens)['compressed_prompt']

if __name__ == "__main__":
    logging.getLogger("transformers").setLevel(logging.ERROR)
    tqdm.pandas()
    # Load the dataset
    dataset_names = ["multifieldqa_en", "hotpotqa", "triviaqa", "narrativeqa"]
    for dataset_name in dataset_names:
        print("Processing dataset:", dataset_name)
        dataset = load_dataset('THUDM/LongBench', dataset_name, split='test')
        df = pd.DataFrame(dataset).iloc[50:100, :]
        ops = df.shape[0]
        compression_rates = [0.1,0.4,0.6,0.9]
        latency_list, memory_list = [],[]
        for i in tqdm(compression_rates, desc=f"Compressing {dataset_name}"):
            monitor = GPUMemoryMonitor(interval=0.1)
            monitor_thread = threading.Thread(target=monitor.start)
            monitor_thread.daemon = True
            monitor_thread.start()

            start_time = time.time()
            df['context_' + str(i)] = df['context'].progress_apply(lambda x: compress_llm(x, rate=i)) # function line
            latency = time.time()- start_time
            latency_list.append(latency/ops)

            monitor.stop()
            monitor_thread.join()
            stats = monitor.get_stats() 
            memory_list.append(stats['avg']/ops)
        #create new dataframe with output latency list and memory list with compression rates as indices

        results = pd.DataFrame(latency_list, index=compression_rates, columns=['latency'])
        results['memory'] = memory_list
        results.to_csv("results_"+dataset_name+'.csv')
        # df = df.sort_index(axis=1)
        # df.to_csv("llmlingua_"+dataset_name+'.csv')

