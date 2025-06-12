import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import json
import glob
from scipy.stats import spearmanr

def load_results_from_json(results_dir='results'):
    """Load all JSON result files and organize by dataset."""
    all_data = {}
    
    # Find all JSON files in results directory
    json_files = glob.glob(os.path.join(results_dir, '*.json'))
    
    for json_file in json_files:
        # Extract dataset name from filename
        # Format: openai_clip-vit-base-patch32_gpt-4o-2024-11-20_DATASET_results.json
        basename = os.path.basename(json_file)
        parts = basename.split('_')
        
        # Handle corner case for CUB_200_2011 dataset
        if 'CUB' in parts and '200' in parts and '2011' in parts:
            dataset_name = 'CUB'
        else:
            dataset_name = parts[-2]  # Extract dataset name before "_results.json"
        
        with open(json_file, 'r') as f:
            data = json.load(f)
            all_data[dataset_name] = data
    
    return all_data

def extract_correlations_by_parameter(all_data, parameter_type='top_k'):
    """Extract correlations for different parameter values."""
    
    if parameter_type == 'top_k':
        param_keys = ['top_k_1', 'top_k_3', 'top_k_5', 'top_k_10', 'top_k_20', 'top_k_50', 'top_k_100', 'top_k_all']
        param_labels = ['1', '3', '5', '10', '20', '50', '100', 'all']
    else:  # num_embed_cutoff
        param_keys = ['num_embed_cutoff_1', 'num_embed_cutoff_3', 'num_embed_cutoff_5', 
                      'num_embed_cutoff_10', 'num_embed_cutoff_20', 'num_embed_cutoff_all']
        param_labels = ['1', '3', '5', '10', '20', 'all']
    
    # Correlation metrics to extract
    metrics = {
        'classification_margin': 'corr_between_classification_margin_scores_and_accuracy',
        'silhouette_scores': 'corr_between_silhouette_scores_and_accuracy',
        'text_only_consistency': 'corr_between_text_only_consistency_scores_and_accuracy',
        'compound_score': 'corr_between_silhouette_scores_plus_text_only_consistency_scores_and_accuracy',
        'pseudo_accuracy': 'corr_between_pseudo_and_real_accuracy'
    }
    
    results = {}
    
    for dataset_name, data in all_data.items():
        results[dataset_name] = {}
        
        for metric_name, metric_key in metrics.items():
            correlations = []
            
            for param_key in param_keys:
                if param_key in data and metric_key in data[param_key]:
                    corr_val = data[param_key][metric_key]
                    correlations.append(corr_val)
                else:
                    correlations.append(np.nan)
            
            results[dataset_name][metric_name] = correlations
    
    return results, param_labels

def plot_parameter_effects(results, param_labels, parameter_type='top_k', metric='classification_margin', save_dir='temporary_exp'):
    """Plot correlation effects across parameter values."""
    
    plt.figure(figsize=(16, 5))
    
    x = np.arange(len(param_labels))
    
    for dataset_name, metrics_data in results.items():
        if metric in metrics_data:
            correlations = metrics_data[metric]
            plt.plot(x, correlations, marker='o', linestyle='-', label=dataset_name)
    
    plt.xticks(x, param_labels, rotation=45)
    plt.xlabel(f'{parameter_type.replace("_", " ").title()} Values')
    plt.ylabel('Spearman ρ (with Zero-Shot Accuracy)')
    
    metric_title = metric.replace('_', ' ').title()
    plt.title(f'{metric_title} Correlations across {parameter_type.replace("_", " ")} values')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    filename = f'{parameter_type}_{metric}_correlations.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

def plot_compound_vs_baseline(results, param_labels, parameter_type='top_k', 
                             baseline_metric='classification_margin', 
                             compound_metric='compound_score', save_dir='temporary_exp'):
    """Plot percentage change from baseline to compound score."""
    
    plt.figure(figsize=(16, 5))
    
    x = np.arange(len(param_labels))
    
    for dataset_name, metrics_data in results.items():
        if baseline_metric in metrics_data and compound_metric in metrics_data:
            baseline = np.array(metrics_data[baseline_metric])
            compound = np.array(metrics_data[compound_metric])
            
            # Calculate percentage change: (compound - baseline) / |baseline| * 100
            pct_change = (compound - baseline) / np.where(baseline != 0, np.abs(baseline), np.nan) * 100
            
            plt.plot(x, pct_change, marker='o', linestyle='--', label=dataset_name)
    
    plt.axhline(0, color='gray', linewidth=1)
    plt.xticks(x, param_labels, rotation=45)
    plt.xlabel(f'{parameter_type.replace("_", " ").title()} Values')
    plt.ylabel('% Change (Compound vs Baseline)')
    
    baseline_title = baseline_metric.replace('_', ' ').title()
    compound_title = compound_metric.replace('_', ' ').title()
    plt.title(f'Percent Change: {compound_title} vs {baseline_title} across {parameter_type.replace("_", " ")} values')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    filename = f'{parameter_type}_percent_change_{compound_metric}_vs_{baseline_metric}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

def plot_multiple_metrics_comparison(results, param_labels, parameter_type='top_k', save_dir='temporary_exp'):
    """Plot multiple metrics on the same chart for comparison."""
    
    if parameter_type == 'top_k':
        metrics_to_compare = ['classification_margin', 'silhouette_scores', 'text_only_consistency', 'compound_score']
    else:  # num_embed_cutoff
        metrics_to_compare = ['classification_margin', 'silhouette_scores', 'text_only_consistency', 'compound_score', 'pseudo_accuracy']
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(metrics_to_compare)))
    
    plt.figure(figsize=(16, 8))
    
    x = np.arange(len(param_labels))
    
    # Calculate average correlation across all datasets for each metric
    avg_correlations = {}
    
    for metric in metrics_to_compare:
        metric_values = []
        for param_idx in range(len(param_labels)):
            values_for_param = []
            for dataset_name, metrics_data in results.items():
                if metric in metrics_data and param_idx < len(metrics_data[metric]):
                    val = metrics_data[metric][param_idx]
                    if not np.isnan(val):
                        values_for_param.append(val)
            
            if values_for_param:
                metric_values.append(np.mean(values_for_param))
            else:
                metric_values.append(np.nan)
        
        avg_correlations[metric] = metric_values
    
    # Plot each metric
    for i, (metric, values) in enumerate(avg_correlations.items()):
        metric_label = metric.replace('_', ' ').title()
        plt.plot(x, values, marker='o', linestyle='-', label=metric_label, 
                color=colors[i], linewidth=2, markersize=8)
    
    plt.xticks(x, param_labels, rotation=45)
    plt.xlabel(f'{parameter_type.replace("_", " ").title()} Values')
    plt.ylabel('Average Spearman ρ (across datasets)')
    plt.title(f'Average Correlation Metrics across {parameter_type.replace("_", " ")} values')
    plt.legend(loc='best', fontsize='medium')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    filename = f'{parameter_type}_multi_metrics_comparison.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

def plot_pseudo_accuracy_analysis(results, param_labels, parameter_type='num_embed_cutoff', save_dir='temporary_exp'):
    """Plot pseudo accuracy analysis specifically for num_embed_cutoff effects."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    x = np.arange(len(param_labels))
    
    # Plot 1: Pseudo accuracy correlation vs real accuracy
    for dataset_name, metrics_data in results.items():
        if 'pseudo_accuracy' in metrics_data:
            correlations = metrics_data['pseudo_accuracy']
            ax1.plot(x, correlations, marker='o', linestyle='-', label=dataset_name)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(param_labels, rotation=45)
    ax1.set_xlabel(f'{parameter_type.replace("_", " ").title()} Values')
    ax1.set_ylabel('Spearman ρ (Pseudo vs Real Accuracy)')
    ax1.set_title('Pseudo Accuracy Correlation with Real Accuracy')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    ax1.grid(alpha=0.3)
    
    # Plot 2: Compare pseudo accuracy correlation with other metrics
    metrics_to_compare = ['pseudo_accuracy', 'classification_margin', 'silhouette_scores']
    colors = ['red', 'blue', 'green']
    
    # Calculate average correlations across datasets
    for i, metric in enumerate(metrics_to_compare):
        avg_correlations = []
        for param_idx in range(len(param_labels)):
            values_for_param = []
            for dataset_name, metrics_data in results.items():
                if metric in metrics_data and param_idx < len(metrics_data[metric]):
                    val = metrics_data[metric][param_idx]
                    if not np.isnan(val):
                        values_for_param.append(val)
            
            if values_for_param:
                avg_correlations.append(np.mean(values_for_param))
            else:
                avg_correlations.append(np.nan)
        
        metric_label = metric.replace('_', ' ').title()
        ax2.plot(x, avg_correlations, marker='o', linestyle='-', label=metric_label, 
                color=colors[i], linewidth=2, markersize=8)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(param_labels, rotation=45)
    ax2.set_xlabel(f'{parameter_type.replace("_", " ").title()} Values')
    ax2.set_ylabel('Average Spearman ρ (across datasets)')
    ax2.set_title('Pseudo Accuracy vs Other Metrics (Average)')
    ax2.legend(loc='best', fontsize='medium')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    filename = f'{parameter_type}_pseudo_accuracy_analysis.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

def main():
    save_dir = 'temporary_exp'
    
    # Load all results
    print("Loading results from JSON files...")
    all_data = load_results_from_json('results')
    print(f"Found data for datasets: {list(all_data.keys())}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving plots to: {save_dir}/")
    
    # Analyze top_k effects
    print("\n=== Analyzing top_k parameter effects ===")
    top_k_results, top_k_labels = extract_correlations_by_parameter(all_data, 'top_k')
    
    # Plot top_k effects for different metrics
    plot_parameter_effects(top_k_results, top_k_labels, 'top_k', 'classification_margin', save_dir)
    plot_parameter_effects(top_k_results, top_k_labels, 'top_k', 'silhouette_scores', save_dir)
    plot_parameter_effects(top_k_results, top_k_labels, 'top_k', 'compound_score', save_dir)
    
    # Plot percentage change for top_k
    plot_compound_vs_baseline(top_k_results, top_k_labels, 'top_k', 
                             'classification_margin', 'compound_score', save_dir)
    
    # Plot multiple metrics comparison for top_k
    plot_multiple_metrics_comparison(top_k_results, top_k_labels, 'top_k', save_dir)
    
    # Analyze num_embed_cutoff effects
    print("\n=== Analyzing num_embed_cutoff parameter effects ===")
    cutoff_results, cutoff_labels = extract_correlations_by_parameter(all_data, 'num_embed_cutoff')
    
    # Plot num_embed_cutoff effects for different metrics
    plot_parameter_effects(cutoff_results, cutoff_labels, 'num_embed_cutoff', 'classification_margin', save_dir)
    plot_parameter_effects(cutoff_results, cutoff_labels, 'num_embed_cutoff', 'silhouette_scores', save_dir)
    plot_parameter_effects(cutoff_results, cutoff_labels, 'num_embed_cutoff', 'compound_score', save_dir)
    plot_parameter_effects(cutoff_results, cutoff_labels, 'num_embed_cutoff', 'pseudo_accuracy', save_dir)
    
    # Plot percentage change for num_embed_cutoff
    plot_compound_vs_baseline(cutoff_results, cutoff_labels, 'num_embed_cutoff', 
                             'classification_margin', 'compound_score', save_dir)
    
    # Plot multiple metrics comparison for num_embed_cutoff
    plot_multiple_metrics_comparison(cutoff_results, cutoff_labels, 'num_embed_cutoff', save_dir)
    
    # Plot pseudo accuracy specific analysis
    plot_pseudo_accuracy_analysis(cutoff_results, cutoff_labels, 'num_embed_cutoff', save_dir)
    
    print(f"\n✅ All plots generated successfully and saved to: {save_dir}/")
    print("\nGenerated files:")
    import glob
    saved_files = glob.glob(os.path.join(save_dir, '*.png'))
    for i, file in enumerate(saved_files, 1):
        print(f"  {i}. {os.path.basename(file)}")
    print(f"\nTotal: {len(saved_files)} plots saved.")

if __name__ == "__main__":
    main() 