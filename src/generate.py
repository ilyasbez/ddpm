import torch
from torchvision.utils import save_image
from IPython.display import Image, display
from ddpm import UNet, LinearNoiseScheduler
import os

os.makedirs("results", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def sample(model, scheduler, num_samples=16, device="cuda"):
    model.eval()
    img_size = 28
    x = torch.randn(num_samples, 1, img_size, img_size, device=device)

    for t in reversed(range(scheduler.timesteps)):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        predicted_noise = model(x, t_batch)

        alpha = scheduler.alphas[t].to(device)
        alpha_cumprod = scheduler.alphas_cumprod[t].to(device)
        beta = scheduler.betas[t].to(device)

        noise = torch.randn_like(x) if t > 0 else 0

        mean = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
        )
        
        posterior_variance = scheduler.posterior_variance[t].to(device)
        x = mean + torch.sqrt(posterior_variance) * noise

    x = (x.clamp(-1, 1) + 1) / 2
    return x

scheduler = LinearNoiseScheduler()
model = UNet(in_channels=1, base_channels=32).to(device)
model.load_state_dict(torch.load("checkpoints/ddpm_mnist.pt", map_location=device))

samples = sample(model, scheduler, num_samples=16, device=device)
save_image(samples, "results/generated_samples.png", nrow=4)

# Show Image in Notebook
display(Image("results/generated_samples.png"))