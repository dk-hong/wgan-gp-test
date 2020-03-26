import os, sys
import matplotlib
import numpy as np

import tflib as lib
import tflib.save_images

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import imageio


sys.path.append(os.getcwd())
matplotlib.use('Agg')

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0

DIM = 64 # Model dimensionality
BATCH_SIZE = 64 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)

lib.print_model_settings(locals().copy())


# ==================Definition Start======================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*DIM, 2*DIM, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM, DIM, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 1, 8, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*DIM, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output.view(-1, OUTPUT_DIM)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
        )
        self.main = main
        self.dc = nn.Linear(4*4*4*DIM, 1)
        self.cl = nn.Sequential(
            nn.Linear(4 * 4 * 4 * DIM, 10),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*DIM)
        d = self.dc(out)
        c = self.cl(out)
        return d, c


def generate_image(frame, netG):
    noise = torch.randn(50, 118)
    for i in range(10):
        if i == 0:
            labels = torch.randint(i, i + 1, (5, 1))
        else:
            labels = torch.cat([labels, torch.randint(i, i + 1, (5, 1))])

    labels_onehot = torch.FloatTensor(50, 10).zero_()
    labels_onehot.scatter_(1, labels, 1)

    if use_cuda:
        noise = noise.cuda(gpu)
        labels_onehot = labels_onehot.cuda(gpu)
    noise = torch.cat([noise, labels_onehot], 1)

    with torch.no_grad():
        noisev = noise
    samples = netG(noisev)
    samples = samples.view(50, 28, 28)

    samples = samples.cpu().data.numpy()
    samples = (samples + 1) / 2

    lib.save_images.save_images(
        samples,
        'tmp/mnist/samples_{}.png'.format(frame)
    )


def visualize_results(epoch):
    netG.eval()

    image_frame_dim = int(np.floor(np.sqrt(100)))

    for i in range(10):
        if i == 0:
            y = torch.randint(i, i + 1, (10, 1))
        else:
            y = torch.cat([y, torch.randint(i, i + 1, (10, 1))])

    sample_y = torch.zeros(10 * 10, 10).scatter_(
        1, y.type(torch.LongTensor), 1)
    sample_z_ = torch.rand((10 * 10, 118))

    if use_cuda:
        sample_z_, sample_y = sample_z_.cuda(), sample_y.cuda()

    samples = netG(torch.cat([sample_z_, sample_y], 1)).view(-1, 1, 28, 28)
    if use_cuda:
        samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
    else:
        samples = samples.data.numpy().transpose(0, 2, 3, 1)

    images = np.squeeze(
        merge(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim]))
    imageio.imwrite('./tmp/by_dataloader/' + '_epoch%03d' % epoch + '.png', images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    real_data = real_data.view(BATCH_SIZE, -1)
    fake_data = fake_data.view(BATCH_SIZE, -1)

    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates.view(-1, 1, 28, 28))[0]

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# ==================Definition End======================


netG = Generator()
netD = Discriminator()

if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.tensor(1.)
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)


CE_loss = nn.CrossEntropyLoss()

d = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))]))
data_loader = torch.utils.data.DataLoader(d, batch_size=64, shuffle=True)

dev_disc_costs = []
for epoch in range(200):
    for idx, (real_data, target) in enumerate(data_loader):
        if idx == data_loader.dataset.__len__() // 64:
            break
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        if use_cuda:
            real_data, target = real_data.cuda(gpu), target.cuda(gpu)
        real_data_v = real_data.view(-1, 1, 28, 28)

        netD.zero_grad()

        # train with real
        D_real, C_real = netD(real_data_v)
        D_real = D_real.mean()
        C_real_loss = CE_loss(C_real, target)
        real_loss = C_real_loss - D_real
        real_loss.backward()

        noise = torch.randn(BATCH_SIZE, 118)

        labels_onehot = torch.FloatTensor(BATCH_SIZE, 10).zero_()

        if use_cuda:
            noise = noise.cuda(gpu)
            labels_onehot = labels_onehot.cuda(gpu)
        labels_onehot.scatter_(1, target.view(BATCH_SIZE, 1), 1)
        noise = torch.cat([noise, labels_onehot], 1)

        with torch.no_grad():
            noisev = noise  # totally freeze netG
        fake = netG(noisev).data
        inputv = fake
        D_fake, C_fake = netD(inputv)
        D_fake = D_fake.mean()
        C_fake_loss = CE_loss(C_fake, target)
        fake_loss = D_fake + C_fake_loss
        fake_loss.backward()

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
        gradient_penalty.backward()

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        optimizerD.step()

        if idx % 5 == 4:
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()

            noise = torch.randn(BATCH_SIZE, 118)
            labels = torch.LongTensor(BATCH_SIZE, 1).random_() % 10

            labels_onehot = torch.FloatTensor(BATCH_SIZE, 10).zero_()
            labels_onehot.scatter_(1, labels, 1)

            if use_cuda:
                noise = noise.cuda(gpu)
                labels = labels.cuda(gpu)
                labels_onehot = labels_onehot.cuda(gpu)
            noise = torch.cat([noise, labels_onehot], 1)
            noisev = noise
            fake = netG(noisev)
            G, C = netD(fake)
            G = G.mean()
            g_loss = CE_loss(C, labels.view(BATCH_SIZE)) - G
            g_loss.backward()
            G_cost = -G
            optimizerG.step()

        if idx % 20 == 19:
            print("Epoch: [%2d] [%4d/%4d] wgan_loss: %.8f, gp: %.8f, G_loss: %.8f" %
                  ((epoch + 1), (idx + 1), data_loader.dataset.__len__() // BATCH_SIZE,
                   (real_loss + fake_loss).item(), gradient_penalty.item(), g_loss.item()))

    # generate_image(epoch, netG)
    with torch.no_grad():
        visualize_results(epoch + 1)

