
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 100
img_size = 64
channels = 3
batch_size = 64
num_epochs = 50
lr = 0.0002
b1 = 0.5
b2 = 0.999

# Image transformations
transforms_ = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load dataset (using MNIST as a placeholder for demonstration)
dataset = datasets.MNIST(
    root="./data/mnist",
    train=True,
    download=True,
    transform=transforms_,
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
)

# Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_size * img_size * channels),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.main(z)
        img = img.view(img.size(0), channels, img_size, img_size)
        return img

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(img_size * img_size * channels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.main(img_flat)
        return validity

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# Loss function
adversarial_loss = nn.BCELoss()

# Training loop
print("
Starting GAN training...")
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # Configure input
        real_imgs = imgs.to(device)
        
        # Adversarial ground truths
        valid = torch.full((real_imgs.size(0), 1), 1.0, device=device)
        fake = torch.full((real_imgs.size(0), 1), 0.0, device=device)

        # -----------------
        #  Train Discriminator
        # -----------------
        optimizer_D.zero_grad()

        # Real images
        real_pred = discriminator(real_imgs)
        d_loss_real = adversarial_loss(real_pred, valid)

        # Fake images
        z = torch.randn(real_imgs.size(0), latent_dim, device=device)
        gen_imgs = generator(z)
        fake_pred = discriminator(gen_imgs.detach())
        d_loss_fake = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        gen_pred = discriminator(gen_imgs)
        g_loss = adversarial_loss(gen_pred, valid)

        g_loss.backward()
        optimizer_G.step()

        if i % 100 == 0:
            print(
                f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                f"D loss: {d_loss.item():.4f} G loss: {g_loss.item():.4f}"
            )

    # Save generated images for inspection
    if epoch % 10 == 0:
        os.makedirs("images", exist_ok=True)
        save_image(gen_imgs.data[:25], f"images/{epoch}.png", nrow=5, normalize=True)

print("GAN training complete.")
