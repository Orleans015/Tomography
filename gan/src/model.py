import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import lightning as L

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

class NaN_Mask(nn.Module):
  def __init__(self):
    super(NaN_Mask, self).__init__()

  def forward(self, x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

class Generator(nn.Module):
    def __init__(self, inputsize):
        super(Generator, self).__init__()
        self.side = 10
        self.linear = nn.Sequential(
            nn.Linear(inputsize, self.side**2 * 1),
            Reshape([-1, 1, self.side, self.side]),
        )
        self.anti_conv = nn.Sequential(
            nn.ConvTranspose2d(1, 5, kernel_size=15, stride=4, padding=0),
            nn.Conv2d(5, 5, kernel_size=11, stride=4, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(5, 10, kernel_size=13, stride=4, padding=0),
            nn.Conv2d(10, 10, kernel_size=9, stride=3, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(10, 20, kernel_size=11, stride=3, padding=0),
            nn.Conv2d(20, 20, kernel_size=9, stride=3, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(20, 10, kernel_size=11, stride=3, padding=0),
            nn.Conv2d(10, 10, kernel_size=7, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(10, 5, kernel_size=9, stride=2, padding=0),
            nn.Conv2d(5, 5, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(5, 1, kernel_size=7, stride=2, padding=0),
            nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=7, stride=2, padding=0),
            nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0),
        )

        self.model = nn.Sequential(
            self.linear,
            self.anti_conv,
        )

    def forward(self, z):
        return self.model(z)
    
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(img_shape[0] * img_shape[1] * img_shape[2], 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class TomoGAN(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.lr = config['learning_rate']
        self.inputsize = config['inputsize']
        self.img_size = config['img_size']
        self.use_device = config['device']
        self.generator = Generator(self.inputsize)
        self.discriminator = Discriminator(self.img_size)
        self.validation_z = torch.randn(8, self.inputsize, device=self.use_device, dtype=torch.float32)
        self.example_input_array = torch.zeros(1, self.inputsize, device=self.use_device, dtype=torch.float32)
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)
  
    def adversarial_loss(self, y_hat, y):
        return nn.BCEWithLogitsLoss()(y_hat, y)

    def training_step(self, batch, batch_idx):
        los, imgs = batch
        z = torch.randn(imgs.size(0), self.inputsize, device=self.use_device).type_as(imgs)

        optimizer_g, optimizer_d = self.optimizers()

        # Train Generator
        self.generated_imgs = self.generator(z)
        valid = torch.ones(imgs.size(0), 1, device=self.use_device).type_as(imgs)
        g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
        self.log('train/g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True)
        optimizer_g.zero_grad()
        self.manual_backward(g_loss)
        optimizer_g.step()

        # Train Discriminator
        valid = torch.ones(imgs.size(0), 1, device=self.use_device).type_as(imgs)
        d_real_loss = self.adversarial_loss(self.discriminator(imgs), valid)
        fake = torch.zeros(imgs.size(0), 1, device=self.use_device).type_as(imgs)
        d_fake_loss = self.adversarial_loss(self.discriminator(self.generated_imgs.detach()), fake)
        d_loss = (d_real_loss + d_fake_loss) / 2
        self.log('train/d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True)
        optimizer_d.zero_grad()
        self.manual_backward(d_loss)
        optimizer_d.step()

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer_g = optim.Adam(self.generator.parameters(), lr=self.lr)
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        self.clip_gradients(optimizer_g, gradient_clip_val=1.0)
        self.clip_gradients(optimizer_d, gradient_clip_val=1.0)
        return [optimizer_g, optimizer_d]

    def on_validation_epoch_end(self):
        self.generated_imgs = self.generator(self.validation_z)
        grid = torchvision.utils.make_grid(self.generated_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)
