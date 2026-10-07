import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ddpm import UNet, LinearNoiseScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Ensure target directory exists for checkpoint saving
os.makedirs("checkpoints", exist_ok=True)

epochs = 15
batch_size = 128
lr = 2e-4
timesteps = 1000

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)
])

dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

scheduler = LinearNoiseScheduler(timesteps=timesteps)

# If LinearNoiseScheduler is an nn.Module, send it to device
if isinstance(scheduler, nn.Module):
    scheduler.to(device)

model = UNet(in_channels=1, base_channels=32).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.MSELoss()

model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for step, (images, _) in enumerate(dataloader):
        images = images.to(device)
        optimizer.zero_grad()

        # Pass t on the appropriate device matching scheduler tensors
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        # Pass t.cpu() because scheduler internal buffers are on CPU
        x_noisy, noise = scheduler.sample_forward(images, t.cpu(), device)
        predicted_noise = model(x_noisy, t)

        loss = criterion(predicted_noise, noise)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.5f}")

torch.save(model.state_dict(), "checkpoints/ddpm_mnist.pt")
print("Training done! Modell saved to 'checkpoints/ddpm_mnist.pt'.")