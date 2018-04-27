# coding=utf-8

"""
Author: angles
Date and time: 27/04/18 - 17:58
"""

import os

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from EmbeddingsImagesDataset import EmbeddingsImagesDataset
from generator_architecture import Generator, weights_init
from utils import create_folder, normalize


class GSN:
    def __init__(self, parameters):
        dir_datasets = os.path.expanduser('~/datasets')
        dir_experiments = os.path.expanduser('~/experiments')

        dataset = parameters['dataset']
        train_attribute = parameters['train_attribute']
        test_attribute = parameters['test_attribute']
        embedding_attribute = parameters['embedding_attribute']

        self.dim = parameters['dim']
        self.nb_channels_first_layer = parameters['nb_channels_first_layer']

        name_experiment = parameters['name_experiment']

        self.dir_x_train = os.path.join(dir_datasets, dataset, '{0}'.format(train_attribute))
        self.dir_x_test = os.path.join(dir_datasets, dataset, '{0}'.format(test_attribute))
        self.dir_z_train = os.path.join(dir_datasets, dataset, '{0}_{1}'.format(train_attribute, embedding_attribute))
        self.dir_z_test = os.path.join(dir_datasets, dataset, '{0}_{1}'.format(test_attribute, embedding_attribute))

        self.dir_experiment = os.path.join(dir_experiments, 'gsn_hf', name_experiment)
        self.dir_models = os.path.join(self.dir_experiment, 'models')
        self.dir_logs = os.path.join(self.dir_experiment, 'logs')
        create_folder(self.dir_models)
        create_folder(self.dir_logs)

        self.batch_size = 128
        self.nb_epochs_to_save = 1

    def train(self, epoch_to_restore=0):
        g = Generator(self.nb_channels_first_layer, self.dim)

        if epoch_to_restore > 0:
            filename_model = os.path.join(self.dir_models, 'epoch_{}.pth'.format(epoch_to_restore))
            g.load_state_dict(torch.load(filename_model))
        else:
            g.apply(weights_init)

        g.cuda()
        g.train()

        dataset = EmbeddingsImagesDataset(self.dir_z_train, self.dir_x_train)
        dataloader = DataLoader(dataset, self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        fixed_dataloader = DataLoader(dataset, 16)
        fixed_batch = next(iter(fixed_dataloader))

        criterion = torch.nn.L1Loss()

        optimizer = optim.Adam(g.parameters())
        writer = SummaryWriter(self.dir_logs)

        try:
            epoch = epoch_to_restore
            while True:
                g.train()
                for _ in range(self.nb_epochs_to_save):
                    epoch += 1

                    for idx_batch, current_batch in enumerate(tqdm(dataloader)):
                        g.zero_grad()
                        x = Variable(current_batch['x']).type(torch.FloatTensor).cuda()
                        z = Variable(current_batch['z']).type(torch.FloatTensor).cuda()
                        g_z = g.forward(z)

                        loss = criterion(g_z, x)
                        loss.backward()
                        optimizer.step()

                    writer.add_scalar('train_loss', loss, epoch)

                z = Variable(fixed_batch['z']).type(torch.FloatTensor).cuda()
                g.eval()
                g_z = g.forward(z)
                images = make_grid(g_z.data[:16], nrow=4, normalize=True)
                writer.add_image('generations', images, epoch)
                filename = os.path.join(self.dir_models, 'epoch_{}.pth'.format(epoch))
                torch.save(g.state_dict(), filename)

        finally:
            print('[*] Closing Writer.')
            writer.close()

    def save_originals(self):
        def _save_originals(dir_z, dir_x, train_test):
            dataset = EmbeddingsImagesDataset(dir_z, dir_x)
            fixed_dataloader = DataLoader(dataset, 16)
            fixed_batch = next(iter(fixed_dataloader))

            temp = make_grid(fixed_batch['x'], nrow=4).numpy().transpose((1, 2, 0))

            filename_images = os.path.join(self.dir_experiment, 'originals_{}.png'.format(train_test))
            Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)

        _save_originals(self.dir_z_train, self.dir_x_train, 'train')
        _save_originals(self.dir_z_test, self.dir_x_test, 'test')

    def compute_errors(self, epoch):
        filename_model = os.path.join(self.dir_models, 'epoch_{}.pth'.format(epoch))
        g = Generator(self.nb_channels_first_layer, self.dim)
        g.cuda()
        g.load_state_dict(torch.load(filename_model))
        g.eval()

        criterion = torch.nn.MSELoss()

        def _compute_error(dir_z, dir_x, train_test):
            dataset = EmbeddingsImagesDataset(dir_z, dir_x)
            dataloader = DataLoader(dataset, batch_size=512, num_workers=4, pin_memory=True)

            error = 0

            for idx_batch, current_batch in enumerate(tqdm(dataloader)):
                x = Variable(current_batch['x']).type(torch.FloatTensor).cuda()
                z = Variable(current_batch['z']).type(torch.FloatTensor).cuda()
                g_z = g.forward(z)

                error += criterion(g_z, x).data.cpu().numpy()

            error /= len(dataloader)

            print('Error for {}: {}'.format(train_test, error))

        _compute_error(self.dir_z_train, self.dir_x_train, 'train')
        _compute_error(self.dir_z_test, self.dir_x_test, 'test')

    def generate_from_model(self, epoch):
        filename_model = os.path.join(self.dir_models, 'epoch_{}.pth'.format(epoch))
        g = Generator(self.nb_channels_first_layer, self.dim)
        g.load_state_dict(torch.load(filename_model))
        g.cuda()
        g.eval()

        def _generate_from_model(dir_z, dir_x, train_test):
            dataset = EmbeddingsImagesDataset(dir_z, dir_x)
            fixed_dataloader = DataLoader(dataset, 16)
            fixed_batch = next(iter(fixed_dataloader))

            z = Variable(fixed_batch['z']).type(torch.FloatTensor).cuda()
            g_z = g.forward(z)
            filename_images = os.path.join(self.dir_experiment, 'epoch_{}_{}.png'.format(epoch, train_test))
            temp = make_grid(g_z.data[:16], nrow=4).cpu().numpy().transpose((1, 2, 0))
            Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)

        _generate_from_model(self.dir_z_train, self.dir_x_train, 'train')
        _generate_from_model(self.dir_z_test, self.dir_x_test, 'test')

        def _generate_path(dir_z, dir_x, train_test):
            dataset = EmbeddingsImagesDataset(dir_z, dir_x)
            fixed_dataloader = DataLoader(dataset, 2, shuffle=True)
            fixed_batch = next(iter(fixed_dataloader))

            z0 = fixed_batch['z'][[0]].numpy()
            z1 = fixed_batch['z'][[1]].numpy()

            batch_z = np.copy(z0)

            nb_samples = 100

            interval = np.linspace(0, 1, nb_samples)
            for t in interval:
                if t > 0:
                    zt = normalize((1 - t) * z0 + t * z1)
                    batch_z = np.vstack((batch_z, zt))

            z = Variable(torch.from_numpy(batch_z)).type(torch.FloatTensor).cuda()
            g_z = g.forward(z)

            # filename_images = os.path.join(self.dir_experiment, 'path_epoch_{}_{}.png'.format(epoch, train_test))
            # temp = make_grid(g_z.data, nrow=nb_samples).cpu().numpy().transpose((1, 2, 0))
            # Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)

            g_z = g_z.data.cpu().numpy().transpose((0, 2, 3, 1))

            folder_to_save = os.path.join(self.dir_experiment, 'epoch_{}_{}_path'.format(epoch, train_test))
            create_folder(folder_to_save)

            for idx in range(nb_samples):
                filename_image = os.path.join(folder_to_save, '{}.png'.format(idx))
                Image.fromarray(np.uint8((g_z[idx] + 1) * 127.5)).save(filename_image)

        _generate_path(self.dir_z_train, self.dir_x_train, 'train')
        _generate_path(self.dir_z_test, self.dir_x_test, 'test')

        def _generate_random():
            nb_samples = 16
            z = np.random.randn(nb_samples, self.dim)
            norms = np.sqrt(np.sum(z ** 2, axis=1))
            norms = np.expand_dims(norms, axis=1)
            norms = np.repeat(norms, self.dim, axis=1)
            z /= norms

            z = Variable(torch.from_numpy(z)).type(torch.FloatTensor).cuda()
            g_z = g.forward(z)
            filename_images = os.path.join(self.dir_experiment, 'epoch_{}_random.png'.format(epoch))
            temp = make_grid(g_z.data[:16], nrow=4).cpu().numpy().transpose((1, 2, 0))
            Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)

        _generate_random()

    def analyze_model(self, epoch):
        filename_model = os.path.join(self.dir_models, 'epoch_{}.pth'.format(epoch))
        g = Generator(self.nb_channels_first_layer, self.dim)
        g.cuda()
        g.load_state_dict(torch.load(filename_model))
        g.eval()

        nb_samples = 50
        batch_z = np.zeros((nb_samples, 32 * self.nb_channels_first_layer, 4, 4))
        # batch_z = np.maximum(5*np.random.randn(nb_samples, 32 * self.nb_channels_first_layer, 4, 4), 0)
        # batch_z = 5 * np.random.randn(nb_samples, 32 * self.nb_channels_first_layer, 4, 4)

        for i in range(4):
            for j in range(4):
                batch_z[:, :, i, j] = create_path(nb_samples)
        # batch_z[:, :, 0, 0] = create_path(nb_samples)
        # batch_z[:, :, 0, 1] = create_path(nb_samples)
        # batch_z[:, :, 1, 0] = create_path(nb_samples)
        # batch_z[:, :, 1, 1] = create_path(nb_samples)
        batch_z = np.maximum(batch_z, 0)

        z = Variable(torch.from_numpy(batch_z)).type(torch.FloatTensor).cuda()
        temp = g.main._modules['4'].forward(z)
        for i in range(5, 10):
            temp = g.main._modules['{}'.format(i)].forward(temp)

        g_z = temp.data.cpu().numpy().transpose((0, 2, 3, 1))

        folder_to_save = os.path.join(self.dir_experiment, 'epoch_{}_path_after_linear_only00_path'.format(epoch))
        create_folder(folder_to_save)

        for idx in range(nb_samples):
            filename_image = os.path.join(folder_to_save, '{}.png'.format(idx))
            Image.fromarray(np.uint8((g_z[idx] + 1) * 127.5)).save(filename_image)


def create_path(nb_samples):
    z0 = 5 * np.random.randn(1, 32 * 32)
    z1 = 5 * np.random.randn(1, 32 * 32)

    # z0 = np.zeros((1, 32 * 32))
    # z1 = np.zeros((1, 32 * 32))

    # z0[0, 0] = -20
    # z1[0, 0] = 20

    batch_z = np.copy(z0)

    interval = np.linspace(0, 1, nb_samples)
    for t in interval:
        if t > 0:
            zt = (1 - t) * z0 + t * z1
            batch_z = np.vstack((batch_z, zt))

    return batch_z
