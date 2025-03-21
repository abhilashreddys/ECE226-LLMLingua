import time
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
def generate_answer(context, question):
    start_time = time.time()  # Start latency timer

    # Generate the answer
    response, _ = model.chat(tokenizer, f"Please read the following text and answer the question below. Context: {context}\n Question: {question}", history=[])
    
    end_time = time.time()  # End latency timer
    
    latency = end_time - start_time  # Calculate latency in seconds
    
    return response, latency

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("data/summarized/multifieldqa_en_summarized.csv")
    monitor = GPUMemoryMonitor(interval=0.1)
    monitor_thread = threading.Thread(target=monitor.start)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Iterate over the dataset
    avg_f1 = 0
    avg_latency = 0
    for index, row in tqdm(df.iterrows()):
        context = row["context"]
        question = row["input"]
        reference = ast.literal_eval(row["answers"])[0]
        
        # Generate the answer
        prediction, latency = generate_answer(context, question)
        
        # Calculate F1 score
        f1 = qa_f1_score(reference, prediction)
        avg_f1 += f1
        avg_latency += latency
        
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
    print(f"Avgerage F1 score: {avg_f1 / len(df)}")
    print(f"Average latency: {avg_latency / len(df)} seconds")