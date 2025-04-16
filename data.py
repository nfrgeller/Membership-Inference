import numpy as np
from datasets import load_dataset, Features, Sequence, Value, ClassLabel
from typing import Dict
import matplotlib.pyplot as plt
import os


def cifar10_scale_and_flatten(example: Dict) -> Dict:
    img = example["img"]
    label = example["label"]
    scaled_flat_image: np.ndarray = np.array(img).flatten() / 255
    return {"img": scaled_flat_image, "label": label}


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("./data/processed", exist_ok=True)
    
    # Load the dataset
    data = load_dataset("uoft-cs/cifar10", cache_dir="./data/raw")
    
    # Select one image to remove (we'll use the first image from the training set)
    image_to_remove = data["train"][0]
    
    # Save the image as PNG
    plt.imsave("removed_image.png", np.array(image_to_remove["img"]))
    
    # Create features for the processed datasets
    cifar10_features = Features(
        {
            "img": Sequence(Value("float32"), length=(32 * 32 * 3)),
            "label": ClassLabel(
                names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], id=None
            ),
        }
    )
    
    # Process the full dataset
    processed_cifar10 = data.map(cifar10_scale_and_flatten, features=cifar10_features)
    
    # Create Cifar_in (full dataset)
    cifar_in = processed_cifar10
    
    # Create Cifar_out (full dataset except the removed image)
    cifar_out = processed_cifar10.filter(lambda x, idx: idx != 0, with_indices=True)
    
    # Save both datasets
    cifar_in.save_to_disk("./data/processed/cifar_in")
    cifar_out.save_to_disk("./data/processed/cifar_out")
