from train import train_model

if __name__ == "__main__":
    # Train 50 models on the in dataset and 50 models on the out dataset.
    for i in range(41, 51):
        train_model(
            model_name=f"sae{i}",
            report_name=f"sae{i}",
            batch_size=500,
            params={"lr": 1e-3, "num_epochs": 75},
            in_dataset=True,
        )
    for i in range(1, 51):
        train_model(
            model_name=f"sae_out{i}",
            report_name=f"sae_out{i}",
            batch_size=500,
            params={"lr": 1e-3, "num_epochs": 75},
            in_dataset=False,
        )
