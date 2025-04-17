import os
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict
from rich.progress import track


class SupervisedAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 3072,
        latent_dim: int = 100,
        hidden_layer_dim: int = 1200,
        pred_layer_dim: int = 50,
        num_classes: int = 10,
        recon_weight: float = 3.0,
        pred_weight: float = 0.1,
        x_key: str = "img",
    ) -> None:
        super().__init__()
        self._pred_layer_dim: int = pred_layer_dim
        self._num_classes: int = num_classes
        self.recon_weight: float = recon_weight
        self.pred_weight: float = pred_weight
        self.classes = [i for i in range(num_classes)]
        self.x_key = x_key

        self.encoder_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, 600),
            nn.ReLU(),
            nn.Linear(600, latent_dim),
        )

        self.decoder_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_dim, 600),
            nn.ReLU(),
            nn.Linear(600, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, input_dim),
        )

        self.prediction_layers = nn.Sequential(
            nn.Linear(latent_dim, pred_layer_dim),
            nn.ReLU(),
            nn.Linear(pred_layer_dim, num_classes),
        )

        self.training_losses: list[torch.tensor] = []
        self.recon_losses: list[float] = []
        self.pred_losses: list[float] = []
        self.validation_losses: list[torch.tensor] = []
        self.validation_recon_losses: list = []
        self.validation_pred_losses: list = []
        self.training_accuracy: float = 0.0
        self.testing_accuracy: list = 0.0

    def encode(self, model_input: torch.tensor) -> torch.tensor:
        return self.encoder_layers(model_input)

    def decode(self, model_input: torch.tensor) -> torch.tensor:
        return self.decoder_layers(model_input)

    def forward(self, model_input: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        latent_rep: torch.tensor = self.encode(model_input)
        reconstruction: torch.tensor = self.decode(latent_rep)
        logits: torch.tensor = self.prediction_layers(latent_rep)
        return reconstruction, logits

    def predict(self, model_input: torch.tensor) -> list[int]:
        self.eval()
        with torch.no_grad():
            if model_input.dim() == 1:
                model_input = model_input.unsqueeze(0)
            _, logits = self(model_input)
            probs = torch.softmax(logits, dim=1)
            predicted_class_indices = torch.argmax(probs, dim=1)
        return [self.classes[id.item()] for id in predicted_class_indices]

    def accuracy(self, data: torch.utils.data.DataLoader) -> float:
        self.eval()
        correct_count = 0
        total_data_count = 0
        with torch.no_grad():
            for d in data:
                X = d[self.x_key]
                y = d["label"]
                predictions = self.predict(X)
                y = y.tolist()
                total_data_count += len(y)
                for i in range(len(y)):
                    if int(y[i]) == int(predictions[i]):
                        correct_count += 1
        return correct_count / total_data_count

    def loss_fn(
        self,
        model_input: torch.tensor,
        label: torch.tensor,
        reconstruction: torch.tensor,
        logits: torch.tensor,
    ) -> torch.tensor:
        reconstruction_mse = nn.MSELoss()
        prediction_loss = nn.CrossEntropyLoss()

        ae_loss = self.recon_weight * reconstruction_mse(model_input, reconstruction)
        pred_loss = self.pred_weight * prediction_loss(logits, label)

        return ae_loss + pred_loss, ae_loss, pred_loss

    def run_training_loop(
        self,
        params: Dict,
        training_data: torch.utils.data.DataLoader,
        validation_data: torch.utils.data.DataLoader,
    ) -> None:
        """
        Run the training loop for a standard autoencoder.

        Parameters
        ----------
        params: Dict
            Dictionary of parameters that should contain `lr` and `num_epochs`.

        training_data: torch.utils.data.DataLoader
            PyTorch dataloader with the training data. Should already be processed and flattened.

        validation_data: torch.utils.data.DataLoader
            PyTorch dataloader with the validation data. Should already be processed and flattened.
        """
        self.training_params = params
        lr = params["lr"]
        num_epochs = params["num_epochs"]
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        N = len(training_data)

        for epoch in range(num_epochs):
            total_loss = 0.0
            total_recon_loss = 0.0
            total_pred_loss = 0.0
            self.train()
            for data in track(training_data):
                image = data[self.x_key]
                label = data["label"]
                reconstruction, logits = self(image)
                loss, recon_loss, pred_loss = self.loss_fn(
                    image, label, reconstruction, logits
                )
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_pred_loss += pred_loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            self.training_losses.append(total_loss / N)
            self.recon_losses.append(total_recon_loss / N)
            self.pred_losses.append(total_pred_loss / N)

            total_loss = 0.0
            total_recon_loss = 0.0
            total_pred_loss = 0.0
            self.eval()
            with torch.no_grad():
                for data in track(validation_data):
                    image = data[self.x_key]
                    label = data["label"]
                    reconstruction, logits = self(image)
                    loss, recon_loss, pred_loss = self.loss_fn(
                        image, label, reconstruction, logits
                    )
                    total_loss += loss.item()
                    total_recon_loss += recon_loss.item()
                    total_pred_loss += pred_loss.item()
                self.validation_losses.append(total_loss / len(validation_data))
                self.validation_recon_losses.append(total_recon_loss / len(validation_data))
                self.validation_pred_losses.append(total_pred_loss / len(validation_data))

        self.training_accuracy = self.accuracy(training_data)
        self.testing_accuracy = self.accuracy(validation_data)

    def plot_and_report(self, report_dir: str) -> None:
        # Set up paths.

        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        report_path = os.path.join(report_dir, "info.txt")
        loss_plot_path = os.path.join(report_dir, "loss_plot.png")
        loss_plot_path2 = os.path.join(report_dir, "loss_plot2.png")

        # Write report to info.txt.
        with open(report_path, "w") as file:
            file.write(f"Training Parameters:{self.training_params}\n")
            file.write(f"Training Accuracy: {self.training_accuracy}\n")
            file.write(f"Testing Accuracy: {self.testing_accuracy}\n")

        # Save loss plot.
        sns.set_theme(style="darkgrid")
        x = [i for i in range(1, len(self.training_losses) + 1)]
        y0 = self.training_losses
        y1 = self.recon_losses
        y2 = self.pred_losses
        df = pd.DataFrame(
            {
                "Epoch": x,
                "Total Loss": y0,
                "Reconstruction Error": y1,
                "Prediction Loss": y2,
            }
        )
        df_melted = pd.melt(
            df, id_vars=["Epoch"], var_name="Loss Type", value_name="Loss Value"
        )

        sns.relplot(
            data=df_melted, x="Epoch", y="Loss Value", hue="Loss Type", kind="line"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.savefig(loss_plot_path)
        plt.close()

        sns.set_theme(style="darkgrid")
        x = [i for i in range(1, len(self.training_losses) + 1)]
        y0 = self.validation_losses
        y1 = self.validation_recon_losses
        y2 = self.validation_pred_losses
        df = pd.DataFrame(
            {
                "Epoch": x,
                "Total Loss": y0,
                "Reconstruction Erorr": y1,
                "Prediction Loss": y2,
            }
        )
        df_melted = pd.melt(
            df, id_vars=["Epoch"], var_name="Loss Type", value_name="Loss Value"
        )

        sns.relplot(
            data=df_melted,
            x="Epoch",
            y="Loss Value",
            hue="Loss Type",
            style="Loss Type",
            kind="line",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Test Loss")
        plt.savefig(loss_plot_path2)

        print(f"Report and plots saved to {report_dir}.")

    def save_model(self, path: str) -> None:
        torch.save(self.state_dict(), path)
