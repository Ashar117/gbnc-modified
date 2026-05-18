#!/usr/bin/env python3
"""GBNC v2 Benchmark Suite.

Runs comprehensive benchmarks on modified GBNC with centrality scoring and 
purity threshold tuning. Tests all combinations of datasets, centrality methods,
and purity thresholds.

Usage:
    python3 benchmark_v2.py                           # Full run (180 jobs)
    python3 benchmark_v2.py --quick                   # Quick test (4 jobs)
    python3 benchmark_v2.py --resume                  # Resume interrupted run
    python3 benchmark_v2.py --datasets cora --runs 5  # Custom config
    
    # Run overnight with screen (recommended):
    screen -S bench
    python3 benchmark_v2.py
    # Press Ctrl+A then D to detach
"""

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Order matters: smaller datasets first (validate before long runs)
DATASETS = ['cora', 'citeseer', 'pubmed', 'cs', 'physics']
METHODS = ['degree', 'degree_centrality', 'closeness', 'betweenness', 'pagerank', 'eigenvector']
PURITIES = [0.7, 0.8, 0.9, 0.95, 0.975, 1.0]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='GBNC v2 Benchmark Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS,
                       help='Datasets to benchmark')
    parser.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS,
                       help='Centrality methods to test')
    parser.add_argument('--purities', nargs='+', type=float, default=PURITIES,
                       help='Purity thresholds to test')
    parser.add_argument('--runs', type=int, default=20,
                       help='Training runs per job (averaged for stability)')
    parser.add_argument('--epochs', type=int, default=60,
                       help='Training epochs per run')
    parser.add_argument('--model', type=str, default='GCN',
                       choices=['GCN', 'APPNP', 'GAT'],
                       help='GNN model architecture')
    parser.add_argument('--output', type=str, default='results_v2.csv',
                       help='Output CSV file for results')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (cora only, 2 methods, 2 purities, 5 runs)')
    parser.add_argument('--resume', action='store_true',
                       help='Skip jobs already in output CSV')
    return parser.parse_args()


def get_completed_jobs(output_file):
    """Read CSV and return set of completed (dataset, method, purity) tuples."""
    completed = set()
    if not os.path.exists(output_file):
        return completed
    
    try:
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('status') == 'success':
                    key = (row['dataset'], row['method'], float(row['purity']))
                    completed.add(key)
    except Exception as e:
        print(f"⚠️  Warning: Could not read existing results: {e}")
    
    return completed


def save_config(args, total_jobs):
    """Save benchmark configuration to JSON for reproducibility."""
    config = {
        'timestamp': datetime.now().isoformat(),
        'datasets': args.datasets,
        'methods': args.methods,
        'purities': args.purities,
        'runs_per_job': args.runs,
        'epochs': args.epochs,
        'model': args.model,
        'total_jobs': total_jobs,
        'output_file': args.output,
    }
    
    config_file = args.output.replace('.csv', '_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_file


def run_job(dataset, method, purity, args):
    """Run a single training job via main.py.
    
    Returns:
        dict with status, elapsed time, and accuracy (if successful)
    """
    cmd = [
        'python3', 'main.py',
        '--dataset', dataset,
        '--score_method', method,
        '--purity_threshold', str(purity),
        '--runs', str(args.runs),
        '--epochs', str(args.epochs),
        '--models', args.model,
    ]
    
    # Create log directory for detailed per-job logs
    log_dir = Path('logs_v2')
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{dataset}_{method}_p{purity:.3f}.log"
    
    start_time = time.time()
    
    try:
        with open(log_file, 'w') as f:
            subprocess.run(
                cmd, 
                stdout=f, 
                stderr=subprocess.STDOUT,
                text=True, 
                timeout=3600  # 1 hour timeout per job
            )
        
        elapsed = time.time() - start_time
        
        # Parse accuracy from log file
        with open(log_file, 'r') as f:
            output = f.read()
        
        ave_acc, std_acc = None, None
        for line in output.split('\n'):
            if 'ave_acc:' in line:
                parts = line.split()
                try:
                    ave_acc = float(parts[1])
                    std_acc = float(parts[3])
                except (ValueError, IndexError):
                    pass
        
        if ave_acc is None:
            return {
                'status': 'failed',
                'elapsed': elapsed,
                'error': 'Could not parse accuracy from output',
                'log_file': str(log_file)
            }
        
        return {
            'status': 'success',
            'elapsed': elapsed,
            'ave_acc': ave_acc,
            'std_acc': std_acc,
            'log_file': str(log_file)
        }
        
    except subprocess.TimeoutExpired:
        return {
            'status': 'timeout',
            'elapsed': time.time() - start_time,
            'error': 'Exceeded 1 hour timeout',
            'log_file': str(log_file)
        }
    except Exception as e:
        return {
            'status': 'error',
            'elapsed': time.time() - start_time,
            'error': str(e),
            'log_file': str(log_file)
        }


def save_result(output_file, dataset, method, purity, result, args):
    """Append a single result to the CSV file."""
    file_exists = os.path.exists(output_file)
    
    with open(output_file, 'a', newline='') as f:
        fieldnames = [
            'timestamp', 'dataset', 'method', 'purity', 'status',
            'elapsed_sec', 'elapsed_min', 'ave_acc', 'std_acc',
            'runs', 'epochs', 'model', 'log_file', 'error'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset,
            'method': method,
            'purity': purity,
            'status': result['status'],
            'elapsed_sec': f"{result['elapsed']:.1f}",
            'elapsed_min': f"{result['elapsed']/60:.2f}",
            'ave_acc': f"{result.get('ave_acc', 0):.4f}" if result.get('ave_acc') else '',
            'std_acc': f"{result.get('std_acc', 0):.4f}" if result.get('std_acc') else '',
            'runs': args.runs,
            'epochs': args.epochs,
            'model': args.model,
            'log_file': result.get('log_file', ''),
            'error': result.get('error', '')
        })


def print_summary(stats, total_jobs, overall_start, output_file):
    """Print final benchmark summary."""
    total_time = time.time() - overall_start
    
    print("\n" + "=" * 80)
    print("🎉 BENCHMARK COMPLETE!")
    print("=" * 80)
    print(f"Total time:      {total_time/3600:.1f} hours ({total_time/60:.1f} min)")
    print(f"Successful:      {stats['success']}/{total_jobs}")
    print(f"Failed:          {stats['failed']}")
    print(f"Timeout:         {stats['timeout']}")
    print(f"Error:           {stats['error']}")
    print(f"Skipped (done):  {stats['skipped']}")
    print(f"Results saved:   {output_file}")
    print(f"Detailed logs:   logs_v2/")
    print("=" * 80)


def main():
    """Run the benchmark suite."""
    args = parse_args()
    
    # Quick test mode override
    if args.quick:
        args.datasets = ['cora']
        args.methods = ['degree', 'closeness']
        args.purities = [0.9, 1.0]
        args.runs = 5
        print("🧪 QUICK TEST MODE\n")
    
    # Validate we're in the right directory
    if not os.path.exists('main.py'):
        print("❌ ERROR: main.py not found. Run this from the gbnc-modified directory!")
        return
    
    # Calculate total jobs
    total_jobs = len(args.datasets) * len(args.methods) * len(args.purities)
    
    # Handle resume mode
    completed = get_completed_jobs(args.output) if args.resume else set()
    
    # Save configuration
    config_file = save_config(args, total_jobs)
    
    # Print header
    print("=" * 80)
    print("GBNC V2 BENCHMARK - MODIFIED VERSION (Centrality + Purity)")
    print("=" * 80)
    print(f"Datasets:        {args.datasets}")
    print(f"Methods:         {args.methods}")
    print(f"Purities:        {args.purities}")
    print(f"Runs per job:    {args.runs}")
    print(f"Epochs:          {args.epochs}")
    print(f"Model:           {args.model}")
    print(f"Total jobs:      {total_jobs}")
    if args.resume:
        print(f"Already done:    {len(completed)} (skipping)")
        print(f"Remaining:       {total_jobs - len(completed)}")
    print(f"Output CSV:      {args.output}")
    print(f"Config saved:    {config_file}")
    print("=" * 80)
    
    # Initialize counters
    stats = {'success': 0, 'failed': 0, 'timeout': 0, 'error': 0, 'skipped': 0}
    job_num = 0
    overall_start = time.time()
    
    # Run all combinations
    for dataset in args.datasets:
        for method in args.methods:
            for purity in args.purities:
                job_num += 1
                job_key = (dataset, method, purity)
                
                # Skip if already completed (resume mode)
                if job_key in completed:
                    print(f"\n⏭️  [{job_num}/{total_jobs}] SKIPPED (already done): "
                          f"{dataset} | {method} | purity={purity}")
                    stats['skipped'] += 1
                    continue
                
                # Run the job
                print(f"\n{'='*80}")
                print(f"🚀 [{job_num}/{total_jobs}] {dataset} | {method} | purity={purity}")
                print(f"{'='*80}")
                
                result = run_job(dataset, method, purity, args)
                save_result(args.output, dataset, method, purity, result, args)
                
                # Update stats and display result
                stats[result['status']] = stats.get(result['status'], 0) + 1
                
                if result['status'] == 'success':
                    print(f"✅ {result['ave_acc']:.4f} ± {result['std_acc']:.4f} "
                          f"({result['elapsed']/60:.1f} min)")
                else:
                    print(f"❌ {result['status'].upper()}: {result.get('error', 'Unknown')}")
                    print(f"   See: {result['log_file']}")
                
                # Show progress and ETA
                jobs_done = sum(stats.values())
                jobs_actual = jobs_done - stats['skipped']
                if jobs_actual > 0:
                    elapsed = time.time() - overall_start
                    avg_time = elapsed / jobs_actual
                    eta_sec = avg_time * (total_jobs - job_num)
                    print(f"📊 Progress: {job_num}/{total_jobs} | "
                          f"✅ {stats['success']} | ❌ {stats['failed'] + stats['timeout'] + stats['error']} | "
                          f"ETA: {eta_sec/3600:.1f}h")
    
    # Final summary
    print_summary(stats, total_jobs, overall_start, args.output)


if __name__ == '__main__':
    main()