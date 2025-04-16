import numpy as np
from datasets import load_dataset, Features, Sequence, Value, ClassLabel
from typing import Dict


def cifar10_scale_and_flatten(example: Dict) -> Dict:
    img = example["img"]
    label = example["label"]
    scaled_flat_image: np.ndarray = np.array(img).flatten() / 255
    return {"img": scaled_flat_image, "label": label}


if __name__ == "__main__":
    data = load_dataset("uoft-cs/cifar10", cache_dir="./data/raw")
    cifar10_features = Features(
        {
            "img": Sequence(Value("float32"), length=(32 * 32 * 3)),
            "label": ClassLabel(
                names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], id=None
            ),
        }
    )
    processed_cifar10 = data.map(cifar10_scale_and_flatten, features=cifar10_features)
    processed_cifar10.save_to_disk("./data/processed/cifar10")
