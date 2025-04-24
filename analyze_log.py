#!/usr/bin/env python3
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

# Set the style for the plots
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

def parse_log_file(log_file_path):
    """Parse the log file to extract token counts and execution times."""
    data = []

    # Regular expressions to match the lines we're interested in
    token_pattern = r"scheduler info: total scheduled requests: \d+, total computing tokens: (\d+)"
    execute_time_pattern = r"execute_model time: ([\d.]+) ms"
    iteration_time_pattern = r"iteration computing time: ([\d.]+) ms"

    current_tokens = None
    current_execute_time = None

    with open(log_file_path, 'r') as f:
        for line in f:
            # Match token count
            token_match = re.search(token_pattern, line)
            if token_match:
                current_tokens = int(token_match.group(1))
                continue

            # Match execute_model time
            execute_time_match = re.search(execute_time_pattern, line)
            if execute_time_match:
                current_execute_time = float(execute_time_match.group(1))
                continue

            # Match iteration computing time and save the complete record
            iteration_time_match = re.search(iteration_time_pattern, line)
            if iteration_time_match and current_tokens is not None and current_execute_time is not None:
                iteration_time = float(iteration_time_match.group(1))
                data.append({
                    'tokens': current_tokens,
                    'execute_model_time': current_execute_time,
                    'iteration_computing_time': iteration_time
                })
                # Reset for next iteration
                current_tokens = None
                current_execute_time = None

    return pd.DataFrame(data)

def create_comparison_plots(df1, df2, log1_name, log2_name, output_prefix):
    """Create plots comparing two log files."""

    # 1. Token Count vs execute_model Time comparison
    plt.figure(figsize=(12, 8))

    # Plot data from first log file
    plt.scatter(df1['tokens'], df1['execute_model_time'], alpha=0.6, color='blue', label=f'{log1_name}')
    z1 = np.polyfit(df1['tokens'], df1['execute_model_time'], 1)
    p1 = np.poly1d(z1)
    plt.plot(df1['tokens'], p1(df1['tokens']), "b--", alpha=0.8)

    # Plot data from second log file
    plt.scatter(df2['tokens'], df2['execute_model_time'], alpha=0.6, color='red', label=f'{log2_name}')
    z2 = np.polyfit(df2['tokens'], df2['execute_model_time'], 1)
    p2 = np.poly1d(z2)
    plt.plot(df2['tokens'], p2(df2['tokens']), "r--", alpha=0.8)

    plt.title('Token Count vs execute_model Time Comparison')
    plt.xlabel('Total Computing Tokens')
    plt.ylabel('execute_model Time (ms)')
    plt.legend()
    plt.grid(True)

    # Add correlation coefficients
    corr1 = df1['tokens'].corr(df1['execute_model_time'])
    corr2 = df2['tokens'].corr(df2['execute_model_time'])
    plt.annotate(f'{log1_name} Correlation: {corr1:.4f}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.3))
    plt.annotate(f'{log2_name} Correlation: {corr2:.4f}',
                xy=(0.05, 0.88), xycoords='axes fraction',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightcoral', alpha=0.3))

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_execute_model_time_comparison.png", dpi=300)
    print(f"Execute model time comparison plot saved as {output_prefix}_execute_model_time_comparison.png")

    # 2. Token Count vs iteration computing time comparison
    plt.figure(figsize=(12, 8))

    # Plot data from first log file
    plt.scatter(df1['tokens'], df1['iteration_computing_time'], alpha=0.6, color='green', label=f'{log1_name}')
    z1 = np.polyfit(df1['tokens'], df1['iteration_computing_time'], 1)
    p1 = np.poly1d(z1)
    plt.plot(df1['tokens'], p1(df1['tokens']), "g--", alpha=0.8)

    # Plot data from second log file
    plt.scatter(df2['tokens'], df2['iteration_computing_time'], alpha=0.6, color='purple', label=f'{log2_name}')
    z2 = np.polyfit(df2['tokens'], df2['iteration_computing_time'], 1)
    p2 = np.poly1d(z2)
    plt.plot(df2['tokens'], p2(df2['tokens']), "m--", alpha=0.8)

    plt.title('Token Count vs Iteration Computing Time Comparison')
    plt.xlabel('Total Computing Tokens')
    plt.ylabel('Iteration Computing Time (ms)')
    plt.legend()
    plt.grid(True)

    # Add correlation coefficients
    corr1 = df1['tokens'].corr(df1['iteration_computing_time'])
    corr2 = df2['tokens'].corr(df2['iteration_computing_time'])
    plt.annotate(f'{log1_name} Correlation: {corr1:.4f}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.3))
    plt.annotate(f'{log2_name} Correlation: {corr2:.4f}',
                xy=(0.05, 0.88), xycoords='axes fraction',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='plum', alpha=0.3))

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_iteration_computing_time_comparison.png", dpi=300)
    print(f"Iteration computing time comparison plot saved as {output_prefix}_iteration_computing_time_comparison.png")

    # 3. Combined comparison plot with all metrics
    plt.figure(figsize=(14, 10))

    # Plot all metrics
    plt.plot(df1['tokens'], df1['execute_model_time'], 'b-', label=f'{log1_name} - Execute Model Time')
    plt.plot(df1['tokens'], df1['iteration_computing_time'], 'g-', label=f'{log1_name} - Iteration Computing Time')
    plt.plot(df2['tokens'], df2['execute_model_time'], 'r-', label=f'{log2_name} - Execute Model Time')
    plt.plot(df2['tokens'], df2['iteration_computing_time'], 'm-', label=f'{log2_name} - Iteration Computing Time')

    plt.title('Token Count vs All Execution Times Comparison')
    plt.xlabel('Total Computing Tokens')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_all_metrics_comparison.png", dpi=300)
    print(f"Combined comparison plot saved as {output_prefix}_all_metrics_comparison.png")

    # 4. Time difference histograms
    plt.figure(figsize=(12, 8))

    # Calculate time differences
    df1['time_difference'] = df1['iteration_computing_time'] - df1['execute_model_time']
    df2['time_difference'] = df2['iteration_computing_time'] - df2['execute_model_time']

    # Plot histograms
    sns.histplot(df1['time_difference'], kde=True, color='blue', alpha=0.5, label=f'{log1_name}')
    sns.histplot(df2['time_difference'], kde=True, color='red', alpha=0.5, label=f'{log2_name}')

    plt.title('Histogram of Time Differences (Iteration - Execute Model)')
    plt.xlabel('Time Difference (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_time_difference_hist_comparison.png", dpi=300)
    print(f"Time difference histogram comparison saved as {output_prefix}_time_difference_hist_comparison.png")

def create_individual_plots(df, log_name, output_prefix):
    """Create plots for a single log file."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Token count vs execute_model time
    ax1.scatter(df['tokens'], df['execute_model_time'], alpha=0.6, color='blue')

    # Add trend line
    z = np.polyfit(df['tokens'], df['execute_model_time'], 1)
    p = np.poly1d(z)
    ax1.plot(df['tokens'], p(df['tokens']), "r--", alpha=0.8)

    ax1.set_title(f'{log_name}: Token Count vs execute_model Time')
    ax1.set_xlabel('Total Computing Tokens')
    ax1.set_ylabel('execute_model Time (ms)')

    # Add correlation coefficient
    corr = df['tokens'].corr(df['execute_model_time'])
    ax1.annotate(f'Correlation: {corr:.4f}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))

    # Plot 2: Token count vs iteration computing time
    ax2.scatter(df['tokens'], df['iteration_computing_time'], alpha=0.6, color='green')

    # Add trend line
    z = np.polyfit(df['tokens'], df['iteration_computing_time'], 1)
    p = np.poly1d(z)
    ax2.plot(df['tokens'], p(df['tokens']), "r--", alpha=0.8)

    ax2.set_title(f'{log_name}: Token Count vs Iteration Computing Time')
    ax2.set_xlabel('Total Computing Tokens')
    ax2.set_ylabel('Iteration Computing Time (ms)')

    # Add correlation coefficient
    corr = df['tokens'].corr(df['iteration_computing_time'])
    ax2.annotate(f'Correlation: {corr:.4f}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_{log_name}_token_vs_time.png", dpi=300)
    print(f"Plot saved as {output_prefix}_{log_name}_token_vs_time.png")

def main():
    parser = argparse.ArgumentParser(description='Analyze and compare log files for token count and execution times.')
    parser.add_argument('log_file1', help='Path to the first log file')
    parser.add_argument('log_file2', help='Path to the second log file')
    parser.add_argument('--output', '-o', default='comparison', help='Output prefix for generated files')

    args = parser.parse_args()

    log_file1 = args.log_file1
    log_file2 = args.log_file2
    output_prefix = args.output

    # Extract log file names for labels
    log1_name = os.path.basename(log_file1).split('.')[0]
    log2_name = os.path.basename(log_file2).split('.')[0]

    print(f"Parsing log file 1: {log_file1}")
    df1 = parse_log_file(log_file1)
    print(f"Found {len(df1)} iterations in {log1_name}")

    print(f"Parsing log file 2: {log_file2}")
    df2 = parse_log_file(log_file2)
    print(f"Found {len(df2)} iterations in {log2_name}")

    # Print summary statistics
    print(f"\nSummary statistics for {log1_name}:")
    print(df1.describe())

    print(f"\nSummary statistics for {log2_name}:")
    print(df2.describe())

    # Save the data to CSV for further analysis if needed
    df1.to_csv(f"{output_prefix}_{log1_name}_data.csv", index=False)
    df2.to_csv(f"{output_prefix}_{log2_name}_data.csv", index=False)
    print(f"Data saved to CSV files")

    # Create individual plots for each log file
    create_individual_plots(df1, log1_name, output_prefix)
    create_individual_plots(df2, log2_name, output_prefix)

    # Create comparison plots
    create_comparison_plots(df1, df2, log1_name, log2_name, output_prefix)

if __name__ == "__main__":
    main()
