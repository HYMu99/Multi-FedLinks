import os
import shutil

def reorganize_data(source_dir, target_dir):
    '''
    This function reorganizes the dataset by splitting the images of each class into training, validation, and test sets
    with a ratio of 3:2:5 (i.e., 30% for training, 20% for validation, and 50% for testing).
    The reorganized dataset will be saved in the specified target directory.
    '''

    modalities = ['FP', 'FV']

    os.makedirs(os.path.join(target_dir, "Process_gray_full_840_2class_325"), exist_ok=True)

    for modality in modalities:

        src_mod_path = os.path.join(source_dir, "Process_gray_full_840_2class", f"{modality}_process_gray_2class")
        dst_mod_path = os.path.join(target_dir, "Process_gray_full_840_2class_325", f"{modality}_process_gray_2class_325")

        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(dst_mod_path, split), exist_ok=True)

        classes = sorted([d for d in os.listdir(src_mod_path) if os.path.isdir(os.path.join(src_mod_path, d))])

        for cls in classes:
            cls_path = os.path.join(src_mod_path, cls)
            images = sorted([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])


            total = len(images)
            train_end = int(total * 0.3)
            val_end = train_end + int(total * 0.2)


            train = images[:train_end]
            val = images[train_end:val_end]
            test = images[val_end:]


            for split in ['train', 'val', 'test']:
                os.makedirs(os.path.join(dst_mod_path, split, cls), exist_ok=True)

            for img in train:
                shutil.copy(os.path.join(cls_path, img),
                           os.path.join(dst_mod_path, "train", cls, img))
            for img in val:
                shutil.copy(os.path.join(cls_path, img),
                           os.path.join(dst_mod_path, "val", cls, img))
            for img in test:
                shutil.copy(os.path.join(cls_path, img),
                           os.path.join(dst_mod_path, "test", cls, img))

    print("\nAll datasets have been reorganized according to the 3:2:5 (train:val:test) split.")

source_directory = "your/dataset/file/path"  # Please specify your dataset file path here
target_directory = "./data_FPV/NUPT-FPV_FULL_325"
reorganize_data(source_directory, target_directory)
