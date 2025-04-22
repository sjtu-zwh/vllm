import re
import pandas as pd
from collections import defaultdict

def parse_log_file(log_file_path):
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # Split content into iterations
    iterations = re.split(r'iteration begin', content)[1:]
    
    results = []
    
    for iter_num, iteration in enumerate(iterations, start=1):
        # Initialize counters for this iteration
        rank_request_counts = defaultdict(int)  # Count of requests per rank
        rank_token_counts = defaultdict(int)    # Sum of tokens per rank
        computing_time = None
        
        # Process each line in the iteration
        lines = iteration.split('\n')
        for line in lines:
            line = line.strip()
            
            # Check for cached or new requests
            if line.startswith('cached request') or line.startswith('new request'):
                # Extract lora rank and scheduled tokens
                rank_match = re.search(r'lora rank: (\d+)', line)
                token_match = re.search(r'scheduled tokens: (\d+)', line)
                
                if rank_match and token_match:
                    rank = int(rank_match.group(1))
                    tokens = int(token_match.group(1))
                    
                    rank_request_counts[rank] += 1
                    rank_token_counts[rank] += tokens
            
            # Check for computing time
            elif line.startswith('iteration computing time'):
                match = re.search(r'iteration computing time: ([\d.]+) ms', line)
                if match:
                    computing_time = float(match.group(1))
        
        # Prepare the result for this iteration
        result = {
            'Iteration': iter_num,
            'Rank 4 Requests': rank_request_counts.get(4, 0),
            '4': rank_token_counts.get(4, 0),
            'Rank 8 Requests': rank_request_counts.get(8, 0),
            '8': rank_token_counts.get(8, 0),
            'Rank 16 Requests': rank_request_counts.get(16, 0),
            '16': rank_token_counts.get(16, 0),
            'Rank 32 Requests': rank_request_counts.get(32, 0),
            '32': rank_token_counts.get(32, 0),
            'Total Requests': sum(rank_request_counts.values()),
            'Total Tokens': sum(rank_token_counts.values()),
            'Computing Time (ms)': computing_time
        }
        if sum(rank_token_counts.values()) != 0:
            results.append(result)
    
    return results

def save_to_csv(results, output_file):
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    columns = ['Iteration',
               '4',
               '8', 
               '16', 
               '32', 
               'Total Tokens',
               'Computing Time (ms)']
    
    df = df[columns]
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# Example usage
log_file_path = '/home/whzhang/workspace/LLM/vllm/logs/20250422_133107.log' # Replace with your log file path
output_csv = 'iteration_stats3.csv'

results = parse_log_file(log_file_path)
save_to_csv(results, output_csv)