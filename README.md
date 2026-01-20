# Neuro-Symbolic Autonomous Driving Framework

This repository contains the implementation of our proposed framework for visual question answering and reasoning in autonomous driving scenarios.

## 📂 Project Structure

  * **`dataset/`**: Contains the processed evaluation splits derived from LingoQA and RobotCar.
  * **`experiments/`**: Source code for running the baseline neural models and our proposed neuro-symbolic experiments.
  * **`reasoningEngine/`**: The symbolic module that references based on the 2PTL (Two-Point Temporal Logic) logic and Horn clause definitions.
  * **`scene_generation/`**: Scripts for generating the Aggregated Scene Description (ASD) from raw inputs.
  * **`vocabulary.json`**: Definition of the closed ontology used for symbolic reasoning.

### Running Experiments

To replicate the main neuro-symbolic results:

1.  **Generate Scene Descriptions:**
    Run the generation scripts in `scene_generation/` to process the input videos.

    ```bash
    python scene_generation/genasd_gemini.py
    ```

2.  **Run Reasoning Engine:**
    Execute the symbolic logic over the generated descriptions.

    ```bash
    python it_check.py --save_it_path [save_path.json] --image_dir [image_dir] --scene [scene_description.json] --rules     [rules.json] -model_name [model_name]
    ```

3.  **Dataset:**
    `dataset/` contains the necessary data split sufficient to run our experiments and reproduce the results.
