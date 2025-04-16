import os
import torch
from datasets import load_from_disk
from model import SupervisedAutoEncoder
import random


def train_model(
    model_name: str,
    report_name: str,
    batch_size=500,
    params={"lr": 1e-3, "num_epochs": 75},
    in_dataset=True,
) -> None:
    # Determine the correct dataset path
    dataset_path = (
        "./data/processed/cifar_in" if in_dataset else "./data/processed/cifar_out"
    )

    # Get data and load it to device.
    data = load_from_disk(dataset_path)
    device = torch.device("mps")

    # Randomly sample half of the dataset
    dataset = data["train"]
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    sampled_indices = indices[: len(indices) // 2]
    sampled_data = dataset.select(sampled_indices)

    # Format the sampled data
    training_data = sampled_data.with_format("torch", device=device)
    testing_data = data["test"].with_format("torch", device=device)

    training_DataLoader = torch.utils.data.DataLoader(
        training_data, batch_size=batch_size, shuffle=True
    )
    testing_DataLoader = torch.utils.data.DataLoader(
        testing_data, batch_size=batch_size, shuffle=True
    )

    # Initialize Model
    model = SupervisedAutoEncoder()
    model.to(device)

    # Train the Model
    model.run_training_loop(params, training_DataLoader, testing_DataLoader)

    # Save Report and Loss Plot
    report_path = os.path.join("./reports", report_name)
    model.plot_and_report(report_path)

    # Save Model
    model_path = os.path.join("./trained_models", model_name + ".pt")
    model.save_model(model_path)


if __name__ == "__main__":
    train_model("sae_test", "sae_test")
