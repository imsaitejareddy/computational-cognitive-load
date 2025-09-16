# Computational Limits in Artificial Intelligence: A Theory of Cognitive Load

This repository contains the official code and data for the research paper, "Computational Limits in Artificial Intelligence: A Theory of Cognitive Load." Our work introduces a formal theory of computational cognitive load and provides the first empirical evidence of a "fragility tipping point" in the reasoning capabilities of modern Large Language Models (LLMs).

## Abstract

The scaling of Large Language Models (LLMs) has exposed a critical gap between performance on static benchmarks and fragility in dynamic, information-rich environments. We introduce a formal theory of **computational cognitive load**, positing that extraneous information (**Context Saturation**) and task-switching (**Attentional Residue**) are key mechanisms that degrade performance. We designed the **Interleaved Cognitive Evaluation (ICE)**, a deconfounded benchmark to systematically manipulate these load factors on a challenging multi-hop reasoning and calculation task. Our large-scale study on a diverse set of SOTA and open-source models revealed a spectrum of cognitive resilience, from the graceful degradation of elite models like Gemini 1.5 Pro to the catastrophic performance collapse of models like Llama 3 under minimal extraneous load.

## Getting Started

### Prerequisites
- Python 3.10+
- An NVIDIA GPU with at least 24GB of VRAM (for running open-source models)
- API keys for OpenAI and Google AI Studio

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/imsaitejareddy/computational-cognitive-load.git](https://github.com/imsaitejareddy/computational-cognitive-load.git)
    cd computational-cognitive-load
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt --extra-index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```

### Running the Experiment

The experiment is controlled via the `run_experiment.py` script.

1.  **Set Environment Variables:** You must set your API keys as environment variables.
    ```bash
    export OPENAI_API_KEY="sk-..."
    export OPENAI_PROJECT_ID="proj_..."
    export GEMINI_API_KEY="..."
    export HF_TOKEN="hf_..."
    ```

2.  **Execute the script:**
    ```bash
    python src/run_experiment.py --num_replications 10 --output_file results/definitive_5_model_results.csv
    ```
    This will run the full 10-replication experiment on all 5 models and save the results to the specified file. This process will take several hours.

### Analyzing the Results

Once the experiment is complete, you can generate the summary tables and plots.

1.  **Execute the analysis script:**
    ```bash
    python src/analyze_results.py --input_file results/definitive_5_model_results.csv --output_plot results/final_plot.png
    ```
    This will read the raw data and generate the final comparison plot in the `results` folder.

## Citing Our Work

If you use this research or code in your work, please cite our paper:
```bibtex
@article{CognitiveLoad2025,
  title={Computational Limits in Artificial Intelligence: A Theory of Cognitive Load},
  author={[Sai Teja Reddy Adapala],
  journal={To Be Determined},
  year={2025}
}
```
