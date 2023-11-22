# python 표준 라이브러리
import os
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

# Hyper-parameters & Variables setting
num_epoch = 200
batch_size = 100
learning_rate = 0.0002
img_size = 28 * 28
num_channel = 1
dir_name = "GAN_results"

noise_size = 100
hidden_size1 = 256
hidden_size2 = 512
hidden_size3 = 1024

# Device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Now using {} devices".format(device))


# Create a directory for saving samples
if not os.path.exists(dir_name):
    os.makedirs(dir_name)


# Dataset transform setting
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)])

# MNIST dataset setting
MNIST_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True,
                                           transform=transform,
                                           download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=MNIST_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(noise_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, img_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.tanh(self.tanh(x))
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(img_size, hidden_size2)
        self.linear2 = nn.Linear(hidden_size2, hidden_size1)
        self.linear3 = nn.Linear(hidden_size1, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.linear1(x))
        x = self.leaky_relu(self.linear2(x))
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x

generator = Generator()
discriminator = Discriminator()

generator = generator.to(device)
discriminator = discriminator.to(device)

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    for i, (images, label) in enumerate(data_loader):

        real_label = torch.full((batch_size, 1), 1, dtype=torch.float32).to(device)
        fake_label = torch.full((batch_size, 1), 0, dtype=torch.float32).to(device)

        real_images = images.reshape(batch_size, -1).to(device)

        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

        ########################
        ## training generator ##
        ########################
        z = torch.randn(batch_size, noise_size).to(device)
        fake_images = generator(z)

        g_loss = criterion(discriminator(fake_images),real_label)

        g_loss.backward()
        g_optimizer.step()

        ############################
        ## training discriminator ##
        ############################
        z = torch.randn(batch_size, noise_size).to(device)
        fake_images = generator(z)
        fake_loss = criterion(discriminator(fake_images), fake_label)
        real_loss = criterion(discriminator(real_images), real_label)
        d_loss = (fake_loss + real_loss) / 2

        d_loss.backward()
        d_optimizer.step()

        d_performance = discriminator(real_images).mean()
        g_performance = discriminator(fake_images).mean()
        
        if (i + 1) % 300 == 0:
            print("Epoch [ {}/{} ]  Step [ {}/{} ]  d_loss : {:.5f}  g_loss : {:.5f}"\
                  .format(epoch, num_epoch, i+1, len(data_loader), d_loss.item(), g_loss.item()))

            print(" Epock {}'s discriminator performance : {:.2f}  generator performance : {:.2f}"\
              .format(epoch, d_performance, g_performance))
        
        samples = fake_images.reshape(batch_size, 1, 28, 28)
        save_image(samples, os.path.join(dir_name, 'GAN_fake_samples{}.png'.format(epoch + 1)))