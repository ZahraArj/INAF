# INAF: INAF Fusion for Sequential Pose Estimation

This repository implements the INAF method for LiDAR pose estimation using sequential fusion and deep learning. The system is designed to work on the KITTI dataset with multiple configuration options and fusion strategies.

---

Project Structure
INAF/
├── main.py
├── Mytools/
│   └── config.yaml
├── Envs/
│   └── environment.yml
├── datasets/
│   └── KITTI/
├── results/
│   └── Kitti_Eval/
├── hyperparam


## Dataset Setup (KITTI Odometry)

To use this project, you **must first download the KITTI Odometry dataset** from the official website:

🔗 **KITTI Odometry Dataset**:  
[http://www.cvlibs.net/datasets/kitti/eval_odometry.php](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)

### Required Folders to Update

After downloading and extracting the dataset, **you must edit the `config.yaml` to match your local paths**:

```yaml
pc_path: '/your/local/path/Kitti/dataset/sequences/'
gt_path:  '/your/local/path/Kitti/dataset/poses/'
output_path: '/your/local/path/INAF/results/'


## Environment Setup

Before running the code, create and activate the required Conda environment:

conda env update -f environment.yml --prune
conda install -c anaconda pytables
conda install -c conda-forge hickle
conda install -c conda-forge pytransform3d
conda install scikit-learn
pip install tensorflow-graphics
pip install keras-tuner==1.3.5
conda activate venv3

## How to Run

Run the main script:
python3 main.py

Make sure to update the config file before executing.


Configuration Guide (config.yaml)

Before running the code, edit the config file to match your desired setup.

Key Configuration Steps:
	•	Set the correct mode in process_number under datasets.kitti.
	•	Set your KITTI data paths:
	•	pc_path
	•	gt_path
	•	output_path
	•	Choose sequences:
	•	For training: sequences_all
	•	For testing: sequences
	•	Set desired model and fusion parameters:
	•	branch_mode, fusion, method, etc.

⸻

Run Modes (process_number)
Mode
Description
1
Save LiDAR and geometry data (preprocess input)
2
Train the model (requires data prepared in Mode 1)
3
Run the trained model (set pre_trained_model to the model folder name)
4
Visualization of model outputs


Workflow
	1.	Prepare input data:
	•	Set process_number: 1
	•	This will preprocess and save data needed for training.
	2.	Train the model:
	•	Set process_number: 2
	•	Tune hyperparameters if needed.
	•	Training outputs will be saved in the saved_model/ directory.
	3.	Run the trained model:
	•	Set process_number: 3
	•	Update pre_trained_model with the folder name of the trained model.
	4.	Visualize results:
	•	Set process_number: 4
	•	Choose the desired pixelvalue under Visualization.