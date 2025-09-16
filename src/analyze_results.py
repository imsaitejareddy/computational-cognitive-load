import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def main(args):
    """
    Analyze the results produced by the Interleaved Cognitive Evaluation (ICE) benchmark.
    
    This function reads a CSV file of raw experiment outputs, computes summary statistics
    (mean and standard error of the mean) for each experimental condition and model,
    reorganizes the data for plotting, and generates a bar chart comparing model
    performance across conditions. The plot and a textual summary are saved/printed.
    """
    # --- Load and Process Data ---
    try:
        df = pd.read_csv(args.input_file)
        print("Successfully loaded results from {}".format(args.input_file))
    except FileNotFoundError:
        print("ERROR: Input file not found at {}".format(args.input_file))
        return

    def clean_model_name(name: str) -> str:
        """Standardize model names for readability."""
        # Update names for new models:
        if 'gpt-4.1' in name:
            return 'GPT-4.1'
        if 'gpt-4o' in name:
            return 'GPT-4o'
        if 'gemini-2.0-flash' in name:
            return 'Gemini 2.0 Flash'
        if 'gemini-1.5-pro' in name:
            return 'Gemini 1.5 Pro'
        if 'Llama-3-70B' in name:
            return 'Llama 3 70B'
        if 'Llama-3-8B' in name:
            return 'Llama 3 8B'
        if 'Mistral-7B' in name:
            return 'Mistral 7B'
        return name

    # Apply cleaning function
    df['model'] = df['model'].apply(clean_model_name)

    def score_accuracy(row) -> bool:
        """Assess whether the model output contains the correct answer."""
        return '11.1' in str(row['output'])

    df['correct'] = df.apply(score_accuracy, axis=1)

    def create_label(row) -> str:
        """Create human-friendly labels for each experimental condition."""
        if row['condition'] == 'ContextSaturation':
            # Combine saturation label without using f-strings to avoid template issues
            return 'Saturation\n(' + str(row['irrelevant_percent']) + '%)'
        if row['condition'] == 'LongContextControl':
            return 'Long Context\n(Control)'
        if row['condition'] == 'AttentionalResidue':
            return 'Attentional\nResidue'
        return row['condition']

    df['label'] = df.apply(create_label, axis=1)

    # --- Statistical Analysis ---
    summary = (
        df.groupby(['label', 'model'])['correct']
        .agg(['mean', 'sem'])
        .reset_index()
    )

    # --- Prepare for Plotting ---
    plot_data = summary.pivot(index='label', columns='model', values=['mean', 'sem'])

    # Define the order of conditions and models for consistent plotting
    condition_order = [
        'Control',
        'Long Context\n(Control)',
        'Attentional\nResidue',
        'Saturation\n(20%)',
        'Saturation\n(50%)',
        'Saturation\n(80%)'
    ]
    model_order = [
        'GPT-4.1',
        'Gemini 2.0 Flash',
        'Llama 3 70B',
        'Llama 3 8B',
        'Mistral 7B'
    ]

    # Reindex to ensure all conditions and models appear even if missing from the data
    plot_data = plot_data.reindex(condition_order)
    # Flatten the MultiIndex columns for easier reordering
    plot_data = (
        plot_data
        .reindex(columns=[(val, mod) for val in ['mean', 'sem'] for mod in model_order], level=[1, 0])
        .reorder_levels([1, 0], axis=1)[model_order]
    )

    # --- Plotting the Definitive Figure ---
    print("Generating final plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    n_models = len(model_order)
    bar_width = 0.15
    index = np.arange(len(condition_order))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, model_name in enumerate(model_order):
        if ('mean', model_name) in plot_data.columns:
            means = plot_data[('mean', model_name)].fillna(0)
            sems = plot_data[('sem', model_name)].fillna(0)
            positions = index + (i - (n_models - 1) / 2) * bar_width
            ax.bar(
                positions,
                means,
                yerr=sems,
                width=bar_width,
                color=colors[i],
                label=model_name,
                capsize=4,
                alpha=0.9
            )

    # Set axis labels and title
    ax.set_xlabel('Experimental Condition', fontsize=16, weight='bold', labelpad=20)
    ax.set_ylabel('Factual Accuracy', fontsize=16, weight='bold', labelpad=20)
    ax.set_title('Cognitive Resilience Across LLM Architectures and Scales', fontsize=22, weight='bold', pad=25)
    ax.set_xticks(index)
    ax.set_xticklabels(condition_order, rotation=45, ha='right', fontsize=14)
    # Generate y-axis tick labels without f-strings
    y_ticks = np.arange(0, 1.2, 0.2)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(int(y * 100)) + '%' for y in y_ticks], fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=14, title='Model', title_fontsize='16')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.grid(axis='x', visible=False)

    plt.tight_layout()
    plt.savefig(args.output_plot)
    print("Plot saved to {}".format(args.output_plot))

    # Print a simple table of mean accuracies
    print("\n--- Definitive 5-Model Accuracy Summary (Mean Accuracy) ---")
    # Round mean values to two decimals and fill missing entries with 'N/A'
    mean_table = plot_data['mean'].round(2).fillna("N/A")
    print(mean_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze results from the ICE benchmark.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file from the experiment.')
    parser.add_argument('--output_plot', type=str, default='results/final_analysis_plot.png', help='Path to save the output plot.')
    args = parser.parse_args()

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    main(args)