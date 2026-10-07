import argparse
import torch
from torchvision.utils import save_image
from ddpm import UNet, LinearNoiseScheduler


@torch.no_grad()
def sample(model, scheduler, num_samples=16, device="cuda"):
    """
    Reverse Denoising Process (Algorithm 2 in Ho et al., 2020)
    """
    model.eval()
    img_size = 28
    x = torch.randn(num_samples, 1, img_size, img_size, device=device)

    for t in reversed(range(scheduler.timesteps)):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = model(x, t_batch)

        alpha = scheduler.alphas[t].to(device)
        alpha_cumprod = scheduler.alphas_cumprod[t].to(device)
        beta = scheduler.betas[t].to(device)

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0

        # Mean calculation
        mean = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
        )
        
        # Reverse step
        x = mean + torch.sqrt(beta) * noise

    # Rescale to [0, 1]
    x = (x.clamp(-1, 1) + 1) / 2
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="checkpoints/ddpm_mnist.pt")
    parser.add_argument("--num_samples", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scheduler = LinearNoiseScheduler()
    model = UNet(in_channels=1, base_channels=32).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    samples = sample(model, scheduler, num_samples=args.num_samples, device=device)
    save_image(samples, "results/generated_samples.png", nrow=4)
    print(f"Generated samples saved to results/generated_samples.png")