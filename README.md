# Variational Autoencoder with U-Net Connections on FaceMask Dataset

## Project Overview

This project implements a Variational Autoencoder (VAE) with U-Net connections using PyTorch, trained on the FaceMask dataset. The goal is to enhance the VAE's performance by integrating U-Net connections, applying annealing schedules for the KL divergence term, and evaluating the model's ability to generate high-fidelity and diverse samples. We also compare the VAE's performance with a standard Autoencoder.

## Dataset

The dataset used for this project is the FaceMask dataset, which contains images of faces with and without masks.

- Dataset Link: [Face Mask 12k Images Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset/data)

## Implementation

### Variational Autoencoder (VAE) with U-Net Connections

The VAE architecture is enhanced with U-Net connections to improve the reconstruction quality. The U-Net connections allow the model to retain more spatial information, which is beneficial for generating high-quality images.

### Annealing Schedules for KL Divergence

To prevent the KL divergence term from dominating the loss function too early in the training process, we implement annealing schedules. This gradually increases the weight of the KL divergence term, allowing the reconstruction loss to have a more significant impact initially.

## Evaluation

### Qualitative and Quantitative Metrics

We evaluate the quality of the generated images using both qualitative and quantitative metrics. The evaluation criteria include:

- Reconstruction ability on input images of the test set.
- Random sampling from the latent space and comparing the outputs with a normal autoencoder.
- t-SNE plot comparison between VAE and normal autoencoder.
- Latent space smoothness evaluation by interpolating between different samples.
- Classification task performance (with or without mask) using latent representations from both VAE and normal autoencoder.

### Results

#### Training and Validation Loss/Accuracy

![Training and Validation Loss/Accuracy Curve](path/to/plot.png)

#### Reconstruction Ability

Comparison of the reconstruction ability of the VAE and a normal autoencoder on input images from the test set.

#### Random Sampling from Latent Space

Comparison of the output from the decoder of the VAE and a normal autoencoder when a point is randomly sampled from the latent space.

#### t-SNE Plot

![t-SNE Plot Comparison](path/to/tsne_plot.png)

#### Latent Space Smoothness

Interpolation between different samples to evaluate the smoothness of the latent space and annealing schedules.

#### Classification Task Performance

Comparison of the performance of latent representations from VAE and a normal autoencoder on the classification task (with or without mask).

## Discussion

### Strategies for Improving VAE's Performance

1. **Data Augmentation**: Increasing the variability in the training data can help the VAE learn more robust features.
2. **More Complex Architectures**: Experimenting with deeper networks or different types of connections (e.g., residual connections) could enhance the model's ability to generate high-fidelity samples.
3. **Better Latent Space Regularization**: Implementing techniques such as beta-VAE or adversarial training could improve the latent space's smoothness and disentanglement.

## Conclusion

This project demonstrates the implementation and evaluation of a VAE with U-Net connections on the FaceMask dataset. The integration of U-Net connections and annealing schedules significantly improves the model's performance. The VAE's ability to generate high-fidelity and diverse samples is compared with a normal autoencoder, and various strategies for further improvement are discussed.

## Requirements

- Python 3.8+
- PyTorch 1.9.0+
- torchvision
- numpy
- pandas
- matplotlib
- scikit-learn
- tqdm

## Usage

1. Clone the repository:

```bash
git clone https://github.com/newArsen/face-mask-vae-unet.git
cd face-mask-vae-unet
