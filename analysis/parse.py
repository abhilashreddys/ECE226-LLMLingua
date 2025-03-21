import re
import pandas as pd
import csv

def parse_logs(log_text):
    # Dictionary to store all the data
    all_data = {}
    
    # Regular expressions to extract data
    dataset_pattern = r'Dataset: (.+?)[\r\n]'
    compression_pattern = r'Compression Ratio ([\d.]+)'
    avg_gpu_pattern = r'Average GPU memory usage: ([\d.]+) MB'
    max_gpu_pattern = r'Maximum GPU memory usage: ([\d.]+) MB'
    avg_f1_pattern = r'Average F1 score: ([\d.]+)'
    avg_latency_pattern = r'Average latency: ([\d.]+) seconds'
    doesnt_exist_pattern = r"Compression Ratio ([\d.]+) doesn't exist"
    
    # Split the log by dataset sections
    dataset_sections = log_text.split('##################################################')
    
    for section in dataset_sections:
        if not section.strip():
            continue
        
        # Extract dataset name
        dataset_match = re.search(dataset_pattern, section)
        if not dataset_match:
            continue
        
        dataset_path = dataset_match.group(1)
        dataset_name = dataset_path.split('/')[-1].replace('.csv', '')
        
        # Determine if it's LLMlingua or random
        method = "llmlingua" if "llmlingua" in dataset_name else "random"
        
        # Clean up dataset name further
        if "llmlingua_" in dataset_name:
            dataset_name = dataset_name.replace("llmlingua_", "")
        if "random_prune_" in dataset_name:
            dataset_name = dataset_name.replace("random_prune_", "")
        
        # Initialize data structure for this dataset if it doesn't exist
        if dataset_name not in all_data:
            all_data[dataset_name] = {"llmlingua": {}, "random": {}}
        
        # Split by compression ratio sections
        compression_sections = section.split('--------------------------------------------------')
        
        for comp_section in compression_sections:
            if not comp_section.strip():
                continue
            
            # Check if this compression ratio exists
            doesnt_exist_match = re.search(doesnt_exist_pattern, comp_section)
            if doesnt_exist_match:
                continue
            
            # Extract compression ratio
            comp_match = re.search(compression_pattern, comp_section)
            if not comp_match:
                continue
            
            comp_ratio = comp_match.group(1)
            
            # Extract metrics
            avg_gpu_match = re.search(avg_gpu_pattern, comp_section)
            max_gpu_match = re.search(max_gpu_pattern, comp_section)
            avg_f1_match = re.search(avg_f1_pattern, comp_section)
            avg_latency_match = re.search(avg_latency_pattern, comp_section)
            
            if avg_gpu_match and max_gpu_match and avg_f1_match and avg_latency_match:
                avg_gpu = float(avg_gpu_match.group(1))
                max_gpu = float(max_gpu_match.group(1))
                avg_f1 = float(avg_f1_match.group(1))
                avg_latency = float(avg_latency_match.group(1))
                
                # Store in data structure
                all_data[dataset_name][method][comp_ratio] = {
                    "avg_gpu": avg_gpu,
                    "max_gpu": max_gpu,
                    "avg_f1": avg_f1,
                    "avg_latency": avg_latency
                }
    
    return all_data

def save_to_csv(data, filename="log_data.csv"):
    # Prepare data for CSV
    rows = []
    
    # Header row
    headers = ["Dataset", "Method", "Compression_Ratio", "Avg_GPU_Memory_MB", 
               "Max_GPU_Memory_MB", "Avg_F1_Score", "Avg_Latency_Sec"]
    rows.append(headers)
    
    # Data rows
    for dataset_name, methods in data.items():
        for method_name, comp_ratios in methods.items():
            for comp_ratio, metrics in comp_ratios.items():
                row = [
                    dataset_name,
                    method_name,
                    comp_ratio,
                    f"{metrics['avg_gpu']:.2f}",
                    f"{metrics['max_gpu']:.2f}",
                    f"{metrics['avg_f1']:.5f}",
                    f"{metrics['avg_latency']:.5f}"
                ]
                rows.append(row)
    
    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)
    
    print(f"Data saved to {filename}")
    return rows

def create_metric_tables(data):
    """Create tables for each metric organized by dataset, method, and compression ratio"""
    
    # Get unique datasets, methods, and compression ratios
    datasets = sorted(list(data.keys()))
    methods = ["llmlingua", "random"]
    comp_ratios = sorted(list(set([cr for dataset in data.values() 
                                   for method in dataset.values() 
                                   for cr in method.keys()])), 
                          key=float)
    
    # Initialize tables for each metric
    tables = {
        "avg_gpu": pd.DataFrame(index=pd.MultiIndex.from_product([methods, datasets]), 
                               columns=comp_ratios),
        "max_gpu": pd.DataFrame(index=pd.MultiIndex.from_product([methods, datasets]), 
                               columns=comp_ratios),
        "avg_f1": pd.DataFrame(index=pd.MultiIndex.from_product([methods, datasets]), 
                              columns=comp_ratios),
        "avg_latency": pd.DataFrame(index=pd.MultiIndex.from_product([methods, datasets]), 
                                   columns=comp_ratios)
    }
    
    # Fill tables with data
    for dataset_name, methods_data in data.items():
        for method_name, comp_data in methods_data.items():
            for comp_ratio, metrics in comp_data.items():
                tables["avg_gpu"].loc[(method_name, dataset_name), comp_ratio] = metrics["avg_gpu"]
                tables["max_gpu"].loc[(method_name, dataset_name), comp_ratio] = metrics["max_gpu"]
                tables["avg_f1"].loc[(method_name, dataset_name), comp_ratio] = metrics["avg_f1"]
                tables["avg_latency"].loc[(method_name, dataset_name), comp_ratio] = metrics["avg_latency"]
    
    return tables

def format_tables(tables):
    """Format tables for display"""
    
    formatted_tables = {}
    
    for metric, table in tables.items():
        # Round numeric values
        if metric in ["avg_gpu", "max_gpu"]:
            formatted = table.round(2)
        elif metric == "avg_f1":
            formatted = table.round(5)
        else:  # avg_latency
            formatted = table.round(3)
        
        # Sort the index for better presentation
        formatted = formatted.sort_index()
        
        formatted_tables[metric] = formatted
    
    return formatted_tables

def main(log_text):
    # Parse log data
    data = parse_logs(log_text)
    
    # Save to CSV
    save_to_csv(data)
    
    # Create tables
    tables = create_metric_tables(data)
    
    # Format tables for display
    formatted_tables = format_tables(tables)
    
    # Print tables
    metric_names = {
        "avg_gpu": "Average GPU Memory Usage (MB)",
        "max_gpu": "Maximum GPU Memory Usage (MB)",
        "avg_f1": "Average F1 Score",
        "avg_latency": "Average Latency (seconds)"
    }
    
    for metric, formatted_table in formatted_tables.items():
        print(f"\n{metric_names[metric]}:")
        print(formatted_table)
        
        # Save individual tables to CSV
        formatted_table.to_csv(f"{metric}_table.csv")
    
    return formatted_tables

# Get log text and process it
# Assuming log_text contains the log data
with open('logs_alldata.txt', 'r') as f:
    log_text = f.read()

# Process the logs
tables = main(log_text)

# Print a nicely formatted markdown table for each metric
for metric, table in tables.items():
    if metric == "avg_gpu":
        metric_name = "Average GPU Memory Usage (MB)"
    elif metric == "max_gpu":
        metric_name = "Maximum GPU Memory Usage (MB)"
    elif metric == "avg_f1":
        metric_name = "Average F1 Score"
    else:  # avg_latency
        metric_name = "Average Latency (seconds)"
    
    print(f"\n## {metric_name}")
    print(table.to_markdown())