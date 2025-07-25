# INAF: INAF Fusion for Sequential Pose Estimation

This repository implements the INAF method for LiDAR pose estimation using sequential fusion and deep learning. The system is designed to work on the KITTI dataset with multiple configuration options and fusion strategies.

---

## Project Structure

- INAF/
  - main.py
  - Mytools/
    - config.yaml
  - Envs/
    - environment.yml
  - datasets/
    - KITTI/
  - results/
    - Kitti_Eval/
  - hyperparam/


## 1. Dataset Setup

To use this project, download the KITTI Odometry dataset from the official website:

http://www.cvlibs.net/datasets/kitti/eval_odometry.php

After extracting the dataset, update the following three paths in the `config.yaml` file to point to your local directories:

```yaml
pc_path: '/your/local/path/Kitti/dataset/sequences/'     # LiDAR point clouds
gt_path: '/your/local/path/Kitti/dataset/poses/'         # Ground-truth poses
output_path: '/your/local/path/INAF/results/'            # Output results
```

## 2. How to Run

Once the config file is ready, run the main script:

```
python3 main.py
```

## 3. Configuration Guide (config.yaml)

Before running, update the configuration file with your specific settings.

### 3.1. Required Fields
	•	process_number: Mode of operation (see Section 5)
	•	pc_path, gt_path, output_path: Local dataset and output paths
	•	sequences_all: List of sequences used for training
	•	sequences: List of sequences used for testing
	•	branch_mode, fusion, method: Model structure and strategy
	•	pre_trained_model: Path to a trained model folder (used in Mode 3)

### 3.2. Changing Branch and Fusion Settings

In the following section of the config file, set your desired branch and fusion strategy:

```
branch_mode: 'geo'       # Options: 'geo', 'lidar', 'all'
fusion: 'soft'           # Options: 'direct', 'soft', 'INAF'
hyper_overwrite: True
data_pre: 'saved_all'
batch_gen: False
calculate_fprimes: False
```

Change branch_mode and fusion here depending on your experiment.

### 3.3. Loading Trained Models

To load a trained model and its hyperparameters (in Mode 3), set:

```
pre_trained_model: '2024_07_27_14_53'
```

Replace the value with the name of your saved model directory.

## 4. Pipeline Modes (process_number)

Mode 1: Save LiDAR and geometry data (preprocessing)
Mode 2: Train the model
Mode 3: Run the trained model (set pre_trained_model to the saved model folder name)
Mode 4: Visualize model predictions

## 5. Workflow

1. Prepare input data:
    - Set process_number: 1
    - This step preprocesses LiDAR and geometry data and saves them to disk

2. Train the model:
    - Set process_number: 2
    - Tune hyperparameters if needed
    - Trained models are saved under saved_model/

3. Run a trained model:
    - Set process_number: 3
    - Set pre_trained_model to the name of the saved model folder

4. Visualize predictions:
    - Set process_number: 4
    - Choose the desired pixelvalue under the Visualization section

## Contact

For questions, please open an issue or contact the repository maintainer.