\# Denoising Diffusion Probabilistic Models (DDPM) from Scratch



A PyTorch implementation of Denoising Diffusion Probabilistic Models based on the seminal paper \[Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239).



This repository contains a full modular pipeline trained on the MNIST dataset, including exact mathematical derivations, a U-Net architecture with time embeddings, and continuous monitoring of noise schedules.



\---



\## 📐 Mathematical Foundations



Denoising Diffusion Probabilistic Models consist of two Markov chains: a \*\*forward process\*\* that incrementally adds Gaussian noise to data, and a \*\*reverse process\*\* where a neural network learns to denoise and recover the data distribution.



\### 1. Forward Process (Diffusion)



Given a data distribution $x\_0 \\sim q(x)$, the forward process adds Gaussian noise according to a variance schedule $\\beta\_1, \\beta\_2, \\dots, \\beta\_T$:



$$q(x\_t \\mid x\_{t-1}) = \\mathcal{N}\\left(x\_t; \\sqrt{1 - \\beta\_t} x\_{t-1}, \\beta\_t \\mathbf{I}\\right)$$



Let $\\alpha\_t = 1 - \\beta\_t$ and $\\bar{\\alpha}\_t = \\prod\_{s=1}^t \\alpha\_s$. Using the reparameterization trick, we can sample $x\_t$ directly at any arbitrary timestep $t$ in closed form without iteratively sampling $x\_1, \\dots, x\_{t-1}$:



$$x\_t = \\sqrt{\\bar{\\alpha}\_t} x\_0 + \\sqrt{1 - \\bar{\\alpha}\_t} \\epsilon, \\quad \\text{where } \\epsilon \\sim \\mathcal{N}(0, \\mathbf{I})$$



\### 2. Reverse Process (Generative Model)



The reverse process parameterizes the joint distribution $p\_\\theta(x\_{0:T})$ as a Markov chain with learned Gaussian transitions starting from standard normal prior $p(x\_T) = \\mathcal{N}(x\_T; 0, \\mathbf{I})$:



$$p\_\\theta(x\_{t-1} \\mid x\_t) = \\mathcal{N}\\left(x\_{t-1}; \\mu\_\\theta(x\_t, t), \\Sigma\_\\theta(x\_t, t)\\right)$$



Ho et al. set $\\Sigma\_\\theta(x\_t, t) = \\sigma\_t^2 \\mathbf{I} = \\beta\_t \\mathbf{I}$ and parameterize the mean $\\mu\_\\theta$ by predicting the added noise $\\epsilon$:



$$\\mu\_\\theta(x\_t, t) = \\frac{1}{\\sqrt{\\alpha\_t}} \\left( x\_t - \\frac{\\beta\_t}{\\sqrt{1 - \\bar{\\alpha}\_t}} \\epsilon\_\\theta(x\_t, t) \\right)$$



\### 3. Objective \& Loss Function



Training optimizes the Variational Lower Bound (VLB). Ho et al. simplified this to an unweighted mean squared error (MSE) objective between the actual noise $\\epsilon$ and the predicted noise $\\epsilon\_\\theta$:



$$\\mathcal{L}\_{\\text{simple}}(\\theta) = \\mathbb{E}\_{t, x\_0, \\epsilon} \\left\[ \\left\\| \\epsilon - \\epsilon\_\\theta\\left( \\sqrt{\\bar{\\alpha}\_t} x\_0 + \\sqrt{1 - \\bar{\\alpha}\_t} \\epsilon, t \\right) \\right\\|^2 \\right]$$



\---



\## 🛠️ Architecture Overview



\- \*\*Time Embedding\*\*: Sinusoidal positional embeddings (similar to Transformers) mapped through two Linear layers with GELU activations.

\- \*\*U-Net Backbone\*\*:

&#x20; - Downsampling \& Upsampling blocks with Residual connections.

&#x20; - Time embedding injected into every residual block via additive conditioning.

\- \*\*Noise Schedule\*\*: Linear variance schedule ranging from $\\beta\_1 = 10^{-4}$ to $\\beta\_T = 0.02$ with $T = 1000$ steps.



\---



\## 🚀 Getting Started



\### Installation

```bash

git clone \[https://github.com/your-username/ddpm-mnist.git](https://github.com/your-username/ddpm-mnist.git)

cd ddpm-mnist

pip install -r requirements.txt

