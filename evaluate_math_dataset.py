"""
Dataset Evaluation Script

This script evaluates MATH or GSM8K dataset problems using various models via Ollama.
It loops through each problem, queries the model, and saves results periodically.
"""

import pandas as pd
import joblib
import os
import re
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from openai import OpenAI


class DatasetEvaluator:
    def __init__(
        self,
        base_url: str = 'http://ollama-server.qmlarki001.qualityminds.de/v1/',
        api_key: str = 'ollama',
        model: str = 'qwen3:8b',
        batch_size: int = 100,
        results_dir: str = 'results',
        dataset: str = 'MATH'
    ):
        """Initialize the dataset evaluator."""
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.results_dir = results_dir
        self.dataset = dataset
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Storage for results
        self.results = []
        
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load the dataset from parquet file."""
        print(f"Loading dataset from: {dataset_path}")
        df = pd.read_parquet(dataset_path)
        print(f"Dataset loaded: {df.shape[0]} problems")
        return df
    
    def create_prompt(self, problem: str) -> str:
        """Create prompt for the model with boxed answer requirement."""
        return f"""{problem}

Please solve this step by step and provide your final answer in a \\boxed{{}} block."""
    
    def extract_boxed_answer(self, response: str) -> Optional[str]:
        """Extract the answer from \\boxed{} format."""
        # Look for \\boxed{...} pattern
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, response)
        
        if matches:
            return matches[-1]  # Return the last boxed answer if multiple exist
        
        # Also try looking for just boxed{...} without double backslash
        pattern = r'boxed\{([^}]*)\}'
        matches = re.findall(pattern, response)
        
        if matches:
            return matches[-1]
            
        return None
    
    def query_model(self, problem: str, max_retries: int = 3) -> Dict:
        """Query the model with retry logic."""
        prompt = self.create_prompt(problem)
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    messages=[
                        {
                            'role': 'user',
                            'content': prompt,
                        }
                    ],
                    model=self.model,
                )
                
                end_time = time.time()
                
                full_response = response.choices[0].message.content
                boxed_answer = self.extract_boxed_answer(full_response)
                response_length = len(full_response) if full_response else 0
                
                return {
                    'success': True,
                    'response': full_response,
                    'boxed_answer': boxed_answer,
                    'response_time': end_time - start_time,
                    'response_length': response_length,
                    'attempt': attempt + 1,
                    'error': None
                }
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return {
                        'success': False,
                        'response': None,
                        'boxed_answer': None,
                        'response_time': None,
                        'response_length': 0,
                        'attempt': attempt + 1,
                        'error': str(e)
                    }
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def save_batch(self, batch_number: int, results: List[Dict]):
        """Save a batch of results using joblib."""
        model_name = self.model.replace(':', '_')
        filename = f"results_batch_{batch_number:04d}_{self.dataset}_{model_name}.joblib"
        filepath = os.path.join(self.results_dir, filename)
        
        joblib.dump(results, filepath)
        print(f"Saved batch {batch_number} ({len(results)} results) to {filepath}")
    
    def evaluate_dataset(self, dataset_path: str, start_idx: int = 0):
        """Evaluate the entire dataset."""
        df = self.load_dataset(dataset_path)
        total_problems = len(df)
        
        print(f"Starting evaluation from index {start_idx}")
        print(f"Model: {self.model}")
        print(f"Batch size: {self.batch_size}")
        print(f"Total problems: {total_problems}")
        print("-" * 50)
        
        start_time = datetime.now()
        
        for i in range(start_idx, total_problems):
            row = df.iloc[i]
            
            if self.dataset == 'MATH':
                print(f"Problem {i+1}/{total_problems}: {row['subject']} Level {row['level']}")
            else:  # GSM8K
                print(f"Problem {i+1}/{total_problems}")
            
            # Query the model
            result = self.query_model(row['problem'])
            
            # Store complete result
            complete_result = {
                'index': i,
                'problem': row['problem'],
                'correct_answer': row['answer'],
                'model_response': result['response'],
                'model_boxed_answer': result['boxed_answer'],
                'success': result['success'],
                'response_time': result['response_time'],
                'response_length': result['response_length'],
                'attempt': result['attempt'],
                'error': result['error'],
                'timestamp': datetime.now().isoformat()
            }
            
            # Add dataset-specific fields
            if self.dataset == 'MATH':
                complete_result.update({
                    'subject': row['subject'],
                    'level': row['level'],
                    'unique_id': row['unique_id']
                })
            # GSM8K doesn't have subject/level/unique_id fields
            
            self.results.append(complete_result)
            
            # Print progress
            if result['success']:
                print(f"  ✓ Response time: {result['response_time']:.2f}s, Length: {result['response_length']} chars")
                if result['boxed_answer']:
                    print(f"  Model answer: {result['boxed_answer']}")
                    print(f"  Correct answer: {row['answer']}")
                else:
                    print(f"  ⚠ No boxed answer found")
            else:
                print(f"  ✗ Failed: {result['error']}")
            
            # Save batch if we've processed enough items
            if (i + 1) % self.batch_size == 0:
                batch_number = (i + 1) // self.batch_size
                batch_results = self.results[-self.batch_size:]
                self.save_batch(batch_number, batch_results)
        
        # Save remaining results
        if len(self.results) % self.batch_size != 0:
            final_batch_number = (len(self.results) // self.batch_size) + 1
            remaining_results = self.results[-(len(self.results) % self.batch_size):]
            self.save_batch(final_batch_number, remaining_results)
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        successful = sum(1 for r in self.results if r['success'])
        total_evaluated = len(self.results)
        
        print("\n" + "="*50)
        print("EVALUATION COMPLETE")
        print("="*50)
        print(f"Total problems evaluated: {total_evaluated}")
        print(f"Successful queries: {successful}")
        print(f"Failed queries: {total_evaluated - successful}")
        print(f"Success rate: {successful/total_evaluated*100:.1f}%")
        print(f"Total time: {duration}")
        print(f"Average time per problem: {duration.total_seconds()/total_evaluated:.2f}s")
        
        # Save final complete results
        model_name = self.model.replace(':', '_')
        final_filename = f"complete_results_{self.dataset}_{model_name}.joblib"
        final_filepath = os.path.join(self.results_dir, final_filename)
        joblib.dump(self.results, final_filepath)
        print(f"Complete results saved to: {final_filepath}")


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate dataset problems using various models via Ollama')
    parser.add_argument('--model', '-m', type=str, default='qwen3:8b',
                        help='Model to use for evaluation (default: qwen3:8b)')
    parser.add_argument('--dataset', '-d', type=str, choices=['MATH', 'GSM8K'], default='MATH',
                        help='Dataset to evaluate (default: MATH)')
    parser.add_argument('--batch-size', '-b', type=int, default=100,
                        help='Batch size for saving results (default: 100)')
    parser.add_argument('--start-idx', '-s', type=int, default=0,
                        help='Starting index for evaluation (default: 0)')
    
    args = parser.parse_args()
    
    # Construct dataset path
    dataset_path = f"data/{args.dataset}/train-00000-of-00001.parquet"
    
    print(f"Starting evaluation with:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Dataset path: {dataset_path}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Start index: {args.start_idx}")
    print("-" * 50)
    
    evaluator = DatasetEvaluator(
        model=args.model,
        dataset=args.dataset,
        batch_size=args.batch_size
    )
    evaluator.evaluate_dataset(dataset_path, start_idx=args.start_idx)


if __name__ == "__main__":
    main()
