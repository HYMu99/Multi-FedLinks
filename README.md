# NUPT-FPV Dataset Preparation and Training

## 1. Install Required Libraries

Before running the code, install all required Python libraries with:

```bash
pip install -r requirements.txt
```


## 2. Original Dataset Structure
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

## 3. Split the Dataset
Open the split.py file and fill in your original dataset path for the source_directory variable.

Then run the following command to split the dataset into training, validation, and test sets:

```bath
python split.py
```
This will generate the processed dataset splits in the target directory specified in the script.

##  4. Train the Model
### Open train.py
### Replace the dataset path in the code with the path to your processed dataset (the output from split.py)
### Run the training script:
```bath
python train.py

```