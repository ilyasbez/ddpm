import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from ddpm import UNet, LinearNoiseScheduler


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Hyperparameters
    epochs = 15
    batch_size = 128
    lr = 2e-4
    timesteps = 200

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)  # Normalize images to [-1, 1]
    ])

    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    scheduler = LinearNoiseScheduler(timesteps=timesteps)
    model = UNet(in_channels=1, base_channels=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for step, (images, _) in enumerate(dataloader):
            images = images.to(device)
            optimizer.zero_grad()

            # Sample random timesteps for each image in batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            # Add noise (Forward diffusion step)
            x_noisy, noise = scheduler.sample_forward(images, t, device)

            # Predict added noise
            predicted_noise = model(x_noisy, t)

            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.5f}")

        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoints/ddpm_mnist.pt")

    print("Training complete. Model saved.")


if __name__ == "__main__":
    train()