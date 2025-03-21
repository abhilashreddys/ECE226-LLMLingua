import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

def load_model_and_tokenizer(model_name):
    """Load the model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation='flash_attention_2',
    )
    return model, tokenizer

def create_compression_prompt(context, compression_ratio):
    """Create a prompt for the model to compress the context"""
    prompt = f"""You are tasked with summarize the following context while preserving its key information. DO NOT COMPRESS TOO MUCH.
    Context: {context}
    
    Compressed context:
    """
    messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt},
    # {"role": "assistant", "content": "Compressed context:"}
    ]
    
    return messages

def compress_context(model, tokenizer, context, compression_ratio=0.5):
    """Compress the given context using the Qwen model based on the compression ratio"""
    messages = create_compression_prompt(context, compression_ratio)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    length = inputs.input_ids.shape[1]
    max_new_tokens = int(compression_ratio * length)
    # print(max_new_tokens)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )
    
    # Extract the generated text (compression result)
    compressed_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return compressed_text

def compress_dataset(compression_ratios=[0.5], model_name = "Qwen/Qwen2.5-7B-Instruct-1M", dataset_name="THUDM/LongBench", dataset_subset="multifieldqa_en", split="test"):
    """
    Compresses the context in the given dataset based on the specified compression ratio
    and returns the dataset with the compressed context column added.
    
    Args:
        compression_ratios: Float between 0 and 1, target compression ratio
        dataset_name: Name of the dataset to load
        dataset_subset: Subset of the dataset to load
        split: Dataset split to use (e.g., "train", "test")
        
    Returns:
        The dataset with an additional column for compressed context
    """
    print(f"Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    print(f"Loading dataset {dataset_name}/{dataset_subset}...")
    ds = load_dataset(dataset_name, dataset_subset)
    
    if split not in ds:
        raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(ds.keys())}")
    
    working_ds = ds[split]  # Use a subset for testing
    
    print(f"Dataset loaded with {len(working_ds)} examples")
    
    # Function to process each example
    def compress_example(example, idx):
        context = example["context"]
        
        # Calculate original token length
        original_tokens = len(tokenizer.encode(context))
        
        # Progress tracking
        # if idx % 10 == 0:
        #     print(f"Processing example {idx}/{len(working_ds)}")
        
        # Compress the context
        for compression_ratio in compression_ratios:
            compressed_context = compress_context(model, tokenizer, context, compression_ratio)
        
            # Calculate compressed token length
            compressed_tokens = len(tokenizer.encode(compressed_context))
        
            # Calculate actual compression ratio
            actual_ratio = compressed_tokens / original_tokens
        
            # Create new fields
            example[f"context_compressed_{compression_ratio}"] = compressed_context
            example[f"compression_ratio_{compression_ratio}_achieved"] = actual_ratio
            example[f"original_tokens_{compression_ratio}"] = original_tokens
            example[f"compressed_tokens_{compression_ratio}"] = compressed_tokens
        
        return example
    
    # Apply the function to each example with index
    processed_ds = working_ds.map(
        lambda example, idx: compress_example(example, idx),
        with_indices=True
    )
    
    for compression_ratio in compression_ratios:
        print("-" * 50)
        print(f"Compression ratio: {compression_ratio:.2f}")
        print("-" * 50)
        # Calculate and print statistics
        original_token_lengths = processed_ds[f"original_tokens_{compression_ratio}"]
        compressed_token_lengths = processed_ds[f"compressed_tokens_{compression_ratio}"]
        actual_ratios = processed_ds[f"compression_ratio_{compression_ratio}_achieved"]
        
        avg_original = sum(original_token_lengths) / len(original_token_lengths)
        avg_compressed = sum(compressed_token_lengths) / len(compressed_token_lengths)
        avg_ratio = sum(actual_ratios) / len(actual_ratios)
        
        print(f"\nCompression Statistics:")
        print(f"  Target ratio: {compression_ratio:.2f}")
        print(f"  Average actual ratio: {avg_ratio:.2f}")
        print(f"  Average original tokens: {avg_original:.1f}")
        print(f"  Average compressed tokens: {avg_compressed:.1f}")
    
    return processed_ds

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compress context in a dataset")
    parser.add_argument("--ratio", type=list, default=[0.25, 0.5, 0.75, 0.99], help="Target compression ratio (0-1)")
    parser.add_argument("--dataset", type=str, default="THUDM/LongBench", help="Dataset name")
    parser.add_argument("--subset", type=str, default="multifieldqa_en", help="Dataset subset")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--output", type=str, default="../data/summarized/", help="Path to save the processed dataset")
    
    args = parser.parse_args()
    
    # Process the dataset
    compressed_ds = compress_dataset(
        compression_ratios=args.ratio,
        dataset_name=args.dataset,
        dataset_subset=args.subset,
        split=args.split
    )
    
    # Save the dataset if requested
    if args.output:
        print(f"\nSaving dataset to {args.output}")
        # compressed_ds.save_to_disk(args.output)
        compressed_ds.to_csv(os.path.join(args.output, f"{args.subset}_summarized.csv"))
        print("Dataset saved successfully")
    
    print("\nDone!")