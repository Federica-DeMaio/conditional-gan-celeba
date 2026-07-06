# Conditional GAN for Face Generation (CelebA) 🎭🤖

This repository contains the development of a **Conditional Generative Adversarial Network (cGAN)** designed to synthesize realistic human faces. The model was trained on the **CelebA** dataset and allows users to control the visual characteristics of the generated images through specific attributes (e.g., *Male/Female, Smiling/Not Smiling, Young/Old*).

## 🎯 Project Objective

The goal of this project is to demonstrate the ability to control the output of a DCGAN-like generative model while overcoming the classic challenges of GAN training (such as *mode collapse* and gradient instability) through advanced architectures and stabilization techniques.

---

## 🧠 Model Architecture (`model.py`)

The system consists of two neural networks competing against each other:

1. **Generator (DCGAN-like):**

   * Takes as input a latent noise vector ($z \in \mathbb{R}^{128}$) concatenated with a multi-hot conditional vector (3 attributes).
   * Uses `ConvTranspose2d` and `BatchNorm2d` blocks to progressively upsample from a 4×4 spatial feature map to a 64×64 RGB image.
   * Final `Tanh` activation maps the output to the range $[-1, 1]$.

2. **Discriminator (Projection Discriminator):**

   * Instead of using the standard concatenation approach, it implements a **Projection Discriminator**. This architecture projects the conditional attributes into the same feature space as the convolutional backbone and computes a dot product.
   * *Advantage:* It evaluates much more accurately and stably whether an image is both real and simultaneously satisfies all the specified conditions, improving the visual quality of multi-attribute queries.

---

## 🛠️ Stabilization Techniques (Training)

Training a GAN requires specific strategies to prevent the Discriminator from becoming too "strong" compared to the Generator. The following best practices have been implemented in `train.py`:

* **TTUR (Two Time-Scale Update Rule):** Different learning rates ($LR_G = 0.0002$, $LR_D = 0.0001$) ensure that the Generator has enough time to learn without being immediately outperformed.
* **Instance Noise with Decay:** Gradually decreasing Gaussian noise is added to the Discriminator inputs during the early training epochs. This "blurs" the initial differences between real and fake samples, stabilizing the gradients.
* **Label Smoothing:** Real targets for the Discriminator are reduced from $1.0$ to $0.9$, decreasing overconfidence and improving gradient flow to the Generator.
* **Perfect Class Balancing (`dataset.py`):** Since the CelebA dataset is highly imbalanced, a `WeightedRandomSampler` assigns weights inversely proportional to the frequency of the 8 possible combinations of the 3 selected attributes, ensuring that the model is trained uniformly across all face categories.

---

## 📂 Repository Structure

* `model.py` - Definition of the `Generator` and `Discriminator` classes in PyTorch.
* `dataset.py` - Image preprocessing, attribute extraction, and balanced dataloader implementation.
* `train.py` - Training pipeline with automatic checkpoint saving, loss visualization, and validation image grids.
* `inference.py` - Optimized script for loading pretrained weights and generating conditional face grids (e.g., "Male, Old, Smiling").

---

## 🚀 Usage (Inference)

To test the model and generate new images, make sure you have downloaded the pretrained weights and placed them in the appropriate directory, then run:

```bash
python inference.py
```

