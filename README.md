# Denoising Diffusion Probabilistic Models (DDPM) from Scratch

A PyTorch implementation of **Denoising Diffusion Probabilistic Models (DDPMs)** based on:

> Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.*  
> [Paper](https://arxiv.org/abs/2006.11239)

This project implements a complete DDPM training and sampling pipeline on the **MNIST** dataset. The implementation includes the forward diffusion process, a U-Net noise predictor with sinusoidal timestep embeddings, noise-schedule construction, training, and image generation.

The goal is to provide a compact implementation that makes the mathematical formulation of DDPMs directly traceable to the corresponding PyTorch code.

---

## Project Structure

```text
ddpm/
├── README.md
├── requirements.txt
│
├── checkpoints/
│   └── ddpm_mnist.pt
│
├── data/
│   └── MNIST/
│       └── raw/
│
├── results/
│
└── src/
    ├── ddpm.py
    ├── train.py
    └── generate.py
```

- `src/ddpm.py` — DDPM model, diffusion process, U-Net components, and noise schedule  
- `src/train.py` — model training  
- `src/generate.py` — image generation using the learned reverse diffusion process  
- `checkpoints/` — saved model weights  
- `data/` — MNIST dataset  
- `results/` — generated samples and training outputs  

The MNIST dataset and generated results are excluded from version control through `.gitignore`.

---

## Mathematical Foundations

A DDPM consists of a forward diffusion process that gradually adds Gaussian noise to data and a learned reverse process that removes this noise.

### 1. Forward Diffusion Process

Given a data sample $x_0 \sim q(x)$, the forward process gradually corrupts it using a variance schedule $\beta_1, \beta_2, \ldots, \beta_T$.

Each transition is defined as:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1-\beta_t}\,x_{t-1}, \beta_t I\right)
$$

Define $\alpha_t = 1-\beta_t$ and:

$$
\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s
$$

The forward process can then be sampled directly at any timestep without explicitly simulating all preceding steps:

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon
$$

where $\epsilon \sim \mathcal{N}(0,I)$. This closed-form expression is used during training to efficiently construct noisy training examples.

### 2. Reverse Diffusion Process

The generative process starts from Gaussian noise $x_T \sim \mathcal{N}(0,I)$ and iteratively generates $x_{T-1}, x_{T-2}, \ldots, x_0$.

The reverse transition is parameterized as:

$$
p_\theta(x_{t-1}\mid x_t) = \mathcal{N}\left(x_{t-1}; \mu_\theta(x_t,t), \Sigma_\theta(x_t,t)\right)
$$

The neural network predicts the noise component $\epsilon_\theta(x_t,t)$. Using this parameterization, the reverse-process mean is:

$$
\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t,t) \right)
$$

Sampling repeatedly applies this reverse transition until an estimate of the original data sample $x_0$ is obtained.

### 3. Training Objective

The original DDPM formulation derives a variational lower bound on the data likelihood. Ho et al. showed that a simplified objective based on noise prediction performs effectively in practice.

The model is trained using the mean squared error between the true noise and the predicted noise:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,x_0,\epsilon} \left[ \left\| \epsilon - \epsilon_\theta(x_t,t) \right\|^2 \right]
$$

During training:

1. A clean MNIST image $x_0$ is sampled.  
2. A random timestep $t$ is selected.  
3. Gaussian noise $\epsilon$ is sampled.  
4. The noisy image $x_t$ is constructed using the closed-form forward process.  
5. The U-Net predicts the noise $\epsilon_\theta(x_t,t)$.  
6. The MSE between the actual and predicted noise is minimized.

---

## Model Architecture

### U-Net

The noise prediction network uses a U-Net-style encoder-decoder architecture. The network contains:

- Downsampling blocks  
- Upsampling blocks  
- Residual connections  
- Skip connections between corresponding encoder and decoder stages  
- Time-dependent conditioning  

The U-Net receives a noisy image $x_t$ together with the corresponding diffusion timestep $t$ and predicts the noise present in the image.

### Time Embeddings

The diffusion timestep is encoded using sinusoidal embeddings. The resulting embedding is processed through fully connected layers and injected into residual blocks, allowing the network to condition its predictions on the current noise level. This conditioning is essential because the denoising problem changes substantially between early and late diffusion timesteps.

### Noise Schedule

The implementation uses a linear variance schedule from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$ with $T = 1000$ diffusion timesteps.

The corresponding quantities $\alpha_t = 1-\beta_t$ and 

$$
\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s
$$

are precomputed and used by both the forward diffusion and reverse sampling procedures.

---

## Installation

Clone the repository:

```bash
git clone [https://github.com/ilyasbez/ddpm.git](https://github.com/ilyasbez/ddpm.git)
cd ddpm
```

A virtual environment is recommended:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Train the DDPM with:

```bash
python src/train.py
```

The MNIST dataset is downloaded automatically if it is not already available locally. The trained model is saved as a checkpoint in the `checkpoints/` directory.

---

## Generating Samples

After training, generate new MNIST samples using:

```bash
python src/generate.py
```

Generated images are written to the `results/` directory. The generation process starts from Gaussian noise and applies the learned reverse diffusion process for $T$ timesteps.

---

## Results

The model learns to transform random Gaussian noise into samples resembling the MNIST training distribution.

Generated samples can be found in `results/`.

Example output:

*(If the generated image has a different filename, update the path above accordingly.)*

---

## Implementation Details

| Component           | Configuration                                      |
|---------------------|----------------------------------------------------|
| Dataset             | MNIST                                              |
| Framework           | PyTorch                                            |
| Diffusion steps     | 1000                                               |
| Noise schedule      | Linear                                             |
| $\beta_1$           | $10^{-4}$                                          |
| $\beta_T$           | $0.02$                                             |
| Prediction target   | Gaussian noise $\epsilon$                          |
| Loss                | Mean Squared Error                                 |
| Architecture        | U-Net                                              |
| Timestep encoding   | Sinusoidal                                         |
| Reference           | Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239) |

---

## License

This project is intended for educational and research purposes.
```
