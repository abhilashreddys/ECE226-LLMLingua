# KV Cache Compression with LLMLingua

## Description
This repository contains the implementation and experiments for our study on **Key-Value (KV) cache compression** along the token dimension using **LLMLingua**, a training-free compression technique for large language models (LLMs). We compare LLMLingua against two baseline approaches: **randomized token pruning** and **summarization-based context compression**. Our experiments are conducted on the **ChatGLM2-6B-32k** model using the **LongBench** dataset, including the `multifieldqa_en`, `hotpotqa`, `narrativeqa`, and `triviaqa` subsets, which focus on long-context question answering. We evaluate the trade-offs between **efficiency** (latency and memory usage) and **accuracy** (F1 score) across varying compression ratios. Our results demonstrate that LLMLingua achieves a better balance between efficiency and accuracy compared to the baselines, particularly at higher compression ratios. This work provides valuable insights into optimizing LLM inference for long-context tasks while minimizing computational overhead.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```

2. Navigate to the project directory:
   ```bash
   cd your-repo-name
   ```

3. Run the setup script to install dependencies:
   ```bash
   ./setup.sh
   ```
