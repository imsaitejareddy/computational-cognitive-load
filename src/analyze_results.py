import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Analyze results from ICE benchmark.
# This script reads CSV results, computes summary statistics,
# prepares data for plotting and generates a bar chart.

def main(args):
    # Load data
    try:
        df = pd.read_csv(args.input_file)
        print("Successfully loaded results from " + str(args.input_file))
    except FileNotFoundError:
        print("ERROR: Input file not found at " + str(args.input_file))
        return

    # Standardize model names
    def clean_model_name(name):
        if 'gpt-4o' in name:
            return 'GPT-4o'
        if 'gemini-1.5-pro' in name:
            return 'Gemini 1.5 Pro'
        if 'Llama-3-70B' in name:
            return 'Llama 3 70B'
        if 'Llama-3-8B' in name:
            return 'Llama 3 8B'
        if 'Mistral-7B' in name:
            return 'Mistral 7B'
        return name

    df['model'] = df['model'].apply(clean_model_name)

    # Compute correctness
    def score_accuracy(row):
        return '11.1' in str(row['output'])
    df['correct'] = df.apply(score_accuracy, axis=1)

    # Create labels for conditions
    def create_label(row):
        if row['condition'] == 'ContextSaturation':
            return 'Saturation\n(' + str(row['irrelevant_percent']) + '%)'
        if row['condition'] == 'LongContextControl':
            return 'Long Context\n(Control)'
        if row['condition'] == 'AttentionalResidue':
            return 'Attentional\nResidue'
        return row['condition']

    df['label'] = df.apply(create_label, axis=1)

    # Statistical summary
    summary = df.groupby(['label','model'])['correct'].agg(['mean','sem']).reset_index()

    # Prepare data
    plot_data = summary.pivot(index='label', columns='model', values=['mean','sem'])
    condition_order = ['Control','Long Context\n(Control)','Attentional\nResidue','Saturation\n(20%)','Saturation\n(50%)','Saturation\n(80%)']
    model_order = ['GPT-4o','Gemini 1.5 Pro','Llama 3 70B','Llama 3 8B','Mistral 7B']
    plot_data = plot_data.reindex(condition_order)
    plot_data = plot_data.reindex(columns=[(v,m) for v in ['mean','sem'] for m in model_order], level=[1,0]).reorder_levels([1,0], axis=1)[model_order]

    # Plot
    print("Generating final plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18,10))
    n_models = len(model_order)
    bar_width = 0.15
    index = np.arange(len(condition_order))
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
    for i, model_name in enumerate(model_order):
        if ('mean', model_name) in plot_data.columns:
            means = plot_data[('mean', model_name)].fillna(0)
            sems = plot_data[('sem', model_name)].fillna(0)
            positions = index + (i - (n_models - 1) / 2) * bar_width
            ax.bar(positions, means, yerr=sems, width=bar_width, color=colors[i], label=model_name, capsize=4, alpha=0.9)

    ax.set_xlabel('Experimental Condition', fontsize=16, weight='bold', labelpad=20)
    ax.set_ylabel('Factual Accuracy', fontsize=16, weight='bold', labelpad=20)
    ax.set_title('Cognitive Resilience Across LLM Architectures and Scales', fontsize=22, weight='bold', pad=25)
    ax.set_xticks(index)
    ax.set_xticklabels(condition_order, rotation=45, ha='right', fontsize=14)
    y_ticks = np.arange(0, 1.2, 0.2)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(int(y*100)) + '%' for y in y_ticks], fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=14, title='Model', title_fontsize='16')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.grid(axis='x', visible=False)

    plt.tight_layout()
    plt.savefig(args.output_plot)
    print("Plot saved to " + str(args.output_plot)

    print("")
    print("--- Definitive 5-Model Accuracy Summary (Mean Accuracy) ---")
    mean_table = plot_data['mean'].round(2).fillna('N/A')
    print(mean_table)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze results from the ICE benchmark.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file from the experiment.')
    parser.add_argument('--output_plot', type=str, default='results/final_analysis_plot.png', help='Path to save the output plot.')
    args = parser.parse_args()
    os.makedirs('results', exist_ok=True)
    main(args)
