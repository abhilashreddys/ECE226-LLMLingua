import time
import os
import re
import ast
import string
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import numpy as np
import threading
from collections import Counter
from tqdm import tqdm

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

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b-32k", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b-32k", trust_remote_code=True)
model = model.half().to("cuda")
model = model.eval()

# Function to calculate F1 score
def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


# Function to generate answers and measure latency/memory
# def generate_answer(context, question):
#     start_time = time.time()  # Start latency timer

#     # Generate the answer
#     response, _ = model.chat(tokenizer, f"Please read the following text and answer the question below. Context: {context}\n Question: {question}", history=[])
    
#     end_time = time.time()  # End latency timer
    
#     latency = end_time - start_time  # Calculate latency in seconds
    
#     return response, latency
def generate_answer(context, question):
    start_time = time.time()  # Start latency timer
    
    # Handle context chunking if too large
    tokens = tokenizer.encode(f"Please read the following text and answer the question below. Context: {context}\n Question: {question}")
    max_tokens = 32000  # Leave room for response
    is_len_exceeded = 0
    
    if len(tokens) > max_tokens:
        is_len_exceeded = 1
        # Tokenize question and instructions separately to preserve them
        question_tokens = tokenizer.encode(f"Please answer the question: {question}")
        question_len = len(question_tokens)
        
        # Calculate available tokens for context
        available_context_tokens = max_tokens - question_len
        
        # Truncate context
        context_tokens = tokenizer.encode(context)
        truncated_context_tokens = context_tokens[:available_context_tokens]
        
        # Decode back to text
        truncated_context = tokenizer.decode(truncated_context_tokens)
        
        # Use the truncated context
        response, _ = model.chat(tokenizer, f"Please read the following text and answer the question below. Context: {truncated_context}\n Question: {question}", history=[])
    else:
        # Use the full context
        response, _ = model.chat(tokenizer, f"Please read the following text and answer the question below. Context: {context}\n Question: {question}", history=[])
    
    end_time = time.time()  # End latency timer
    latency = end_time - start_time  # Calculate latency in seconds
    
    return response, latency, is_len_exceeded

def run_inference(ratio=25):
    monitor = GPUMemoryMonitor(interval=0.1)
    monitor_thread = threading.Thread(target=monitor.start)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Iterate over the dataset
    avg_f1 = 0
    avg_latency = 0
    num_len_exceeded = 0
    avg_f1_no_len_exceeded = 0
    avg_latency_no_len_exceeded = 0
    for index, row in df.iterrows():
        context = row[f"context_{ratio}"]
        
        question = row["input"]
        reference = ast.literal_eval(row["answers"])[0]
        
        # Generate the answer
        prediction, latency, len_exceeded = generate_answer(context, question)
        num_len_exceeded += len_exceeded
        
        # Calculate F1 score
        f1 = qa_f1_score(reference, prediction)
        avg_f1 += f1
        avg_latency += latency
        
        if len_exceeded == 0:
            avg_f1_no_len_exceeded += f1
            avg_latency_no_len_exceeded += latency
        
        # Print the results
        # print(f"Context: {context}")
        # print(f"Question: {question}")
        # print(f"Reference: {reference}")
        # print(f"Prediction: {prediction}")
        # print(f"F1 Score: {f1}")
        # print(f"Latency: {latency} seconds")
        # print("--------------------------------------------------")
        # break
    
    monitor.stop()
    monitor_thread.join()
    # Get statistics
    stats = monitor.get_stats()
    print(f"Average GPU memory usage: {stats['avg']:.2f} MB")
    print(f"Maximum GPU memory usage: {stats['max']:.2f} MB")
    print(f"Average F1 score: {avg_f1 / len(df)}")
    print(f"Average latency: {avg_latency / len(df)} seconds")
    print(f"Number of times length exceeded: {num_len_exceeded}")
    print(f"Average F1 score (no length exceeded): {avg_f1_no_len_exceeded / (len(df) - num_len_exceeded)}")
    print(f"Average latency (no length exceeded): {avg_latency_no_len_exceeded / (len(df) - num_len_exceeded)} seconds")
    print("--------------------------------------------------")
if __name__ == "__main__":
    # Load the dataset
    # dataset_paths = [os.path.join("data/llmlingua2", f) for f in os.listdir("data/llmlingua2") if os.path.isfile(os.path.join("data/llmlingua2", f))]
    # dataset_paths += [os.path.join("data/random2", f) for f in os.listdir("data/random2") if os.path.isfile(os.path.join("data/random2", f))]
    dataset_paths = ["data/random2/random_prune_hotpotqa.csv"]
    print("datasets:", dataset_paths)
    compression_ratios = [0.75, 0.9] #[0.1, 0.25, 0.4, 0.5, 0.6, 
    # compression_ratios = [0.6]
    for data_path in dataset_paths:
        print("#"*50)
        print("Dataset:", data_path)
        df = pd.read_csv(data_path)
        for cr in compression_ratios:
            if f"context_{cr}" in df.columns:
                print("Compression Ratio {}".format(cr))
                try:
                    run_inference(cr)
                except Exception as e:
                    print("Error in processing Compression Ratio {}".format(cr))
                    print(e)
            else:
                print("Compression Ratio {} doesn't exist".format(cr))
    # print("Compression Ratio {}".format(50))
    # run_inference(0.5)
    # print("Compression Ratio {}".format(75))
    # run_inference(0.75)