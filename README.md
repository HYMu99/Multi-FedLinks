# NUPT-FPV Dataset Preparation and Training

## 1. Install Required Libraries

Before running the code, install all required Python libraries with:

```bash
pip install -r requirements.txt
```

## Creating Folders in the Project Root Directory

To set up the project, create the following folders in the root directory:

1. `pair`
2. `c_pth`
3. `federated_data`
4. `data_FPV`

You can create these folders using the following command in your terminal:

```bash
mkdir pair c_pth federated_data data_FPV
```

## 3. Original Dataset Structure
Please organize your original dataset as shown below:


```bath
data_FPV/
└── NUPT-FPV_FULL/
    ├── Process_gray_full_840_1class/
    │   ├── FP_process_gray_1class/
    │   └── FV_process_gray_1class/
    └── Process_gray_full_840_2class/
        ├── FP_process_gray_2class/
        └── FV_process_gray_2class/
```

## 4. Split the Dataset
Open the split.py file and fill in your original dataset path for the source_directory variable.

Then run the following command to split the dataset into training, validation, and test sets:

```bath
python split.py
```
This will generate the processed dataset splits in the target directory specified in the script.

## 5. Train the Model
### Open train.py
### Replace the dataset path in the code with the path to your processed dataset (the output from split.py)
### Run the training script:
```bath
python train.py
```

