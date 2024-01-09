# https://avandekleut.github.io/vae/
import matplotlib.pyplot as plt
import torch.utils
import torchvision

from germina.vae import Autoencoder, train, plot_latent, plot_reconstructed, VariationalAutoencoder, train2

torch.manual_seed(0)

plt.rcParams['figure.dpi'] = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

latent_dims = 2
# autoencoder = Autoencoder(latent_dims).to(device)  # GPU
autoencoder = Autoencoder(latent_dims)

data = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', transform=torchvision.transforms.ToTensor(), download=True), batch_size=128, shuffle=True)
autoencoder = train(autoencoder, data)

plot_latent(autoencoder, data)
plot_reconstructed(autoencoder)

# vae = VariationalAutoencoder(latent_dims).to(device)  # GPU
vae = VariationalAutoencoder(latent_dims)
vae = train2(vae, data)

plot_latent(vae, data)
plot_reconstructed(vae, r0=(-3, 3), r1=(-3, 3))
