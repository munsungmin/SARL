# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the AIME dataset (1983-2024) CSV to parquet format
"""

import argparse
import json
import os

import pandas as pd
from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="./data/AIME_Dataset_1983_2024.csv", 
                        help="Path to the AIME CSV file")
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_save_dir", default="./data/aime", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()

    data_source = "AIME_1983_2024"
    print(f"Loading AIME dataset from {args.csv_path}...", flush=True)
    
    # Read CSV
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} problems from AIME dataset")
    
    # Show column names for debugging
    print(f"Columns: {df.columns.tolist()}")
    
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # Process data
    processed_data = []
    for idx, row in df.iterrows():
        question = row["Question"]
        answer = str(row["Answer"]).strip()
        year = row["Year"]
        problem_num = row["Problem Number"]
        problem_id = row["ID"]
        
        # Add instruction to question
        question_with_instruction = question + " " + instruction_following
        
        data = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question_with_instruction}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "train", 
                "index": idx,
                "year": int(year),
                "problem_number": int(problem_num),
                "problem_id": problem_id
            },
        }
        processed_data.append(data)
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_list(processed_data)
    print(f"Created dataset with {len(train_dataset)} examples")

    # Determine save directory
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    hdfs_dir = args.hdfs_dir
    
    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Save to parquet
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    print(f"Saved train.parquet to {local_dir}")
    
    # Save one example as JSON for reference
    example = train_dataset[0]
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2)
    print(f"Saved train_example.json to {local_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("AIME Dataset Summary")
    print("="*60)
    print(f"Total problems: {len(train_dataset)}")
    years = df["Year"].unique()
    print(f"Years covered: {min(years)} - {max(years)}")
    print(f"Output directory: {local_dir}")
    print("="*60)
    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

