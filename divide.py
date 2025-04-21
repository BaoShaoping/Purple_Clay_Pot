import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(input_dir, output_dir, ratios=(0.7, 0.15, 0.15), seed=42):
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    classes = os.listdir(input_dir)
    for cls in classes:
        cls_path = os.path.join(input_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        images = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        labels = [cls] * len(images)

        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=ratios[2], stratify=labels, random_state=seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=ratios[1]/(ratios[0]+ratios[1]), stratify=y_train, random_state=seed
        )

        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

        def copy_files(file_list, dest_dir):
            for src in file_list:
                dest = os.path.join(dest_dir, cls, os.path.basename(src))
                shutil.copy(src, dest)

        copy_files(X_train, train_dir)
        copy_files(X_val, val_dir)
        copy_files(X_test, test_dir)

if __name__ == "__main__":
    split_dataset(
        input_dir="./augmented_data",
        output_dir="./split_dataset",
        ratios=(0.7, 0.15, 0.15),
        seed=42
    )