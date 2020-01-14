import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from models1 import LSTMDiscriminator, LSTMGenerator
from models2 import Discriminator, Generator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=200,
                        help="number of epochs of training")
    parser.add_argument("--batch", type=int, default=128,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--n_cpu", type=int, default=0,
                        help="number of cpu threads to use during batch generation")

    parser.add_argument("--input_dim", type=int, default=300,
                        help="dimensionality of the input")
    parser.add_argument("--latent_dim", type=int, default=256,
                        help="dimensionality of the latent")

    parser.add_argument("--sample_interval", type=int, default=100,
                        help="interval betwen image samples")
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='if use CUDA')
    args_ = parser.parse_args()
    args_.device = torch.device('cuda' if args_.use_cuda and torch.cuda.is_available() else 'cpu')
    print(args_)
    return args_


class PPGDataSet(Dataset):
    def __init__(self, arr):
        self.dim = arr.shape[1]
        self.data = torch.as_tensor(arr.astype(np.float32))

    def __getitem__(self, index):
        out = self.data[index].reshape(-1, self.dim)
        # out = self.data[index]
        return out

    def __len__(self):
        return self.data.shape[0]


def main():
    os.makedirs("gen_signal", exist_ok=True)

    fp = 'nsrdb_rri.pickle'
    df = pd.read_pickle(fp)
    rri = np.vstack(df['rri'].values * 1000)
    train_rri = rri.reshape(rri.shape[0], rri.shape[1], 1)
    print('train rri shape: ', train_rri.shape)

    train_set = PPGDataSet(train_rri)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.n_cpu)

    input_shape = (1, args.input_dim)

    # Input: noise of shape (batch_size, seq_len, in_dim)
    # Output: sequence of shape (batch_size, seq_len, out_dim)
    # generator = LSTMGenerator(input_shape).to(args.device)
    generator = Generator(input_shape, args.latent_dim).to(args.device)

    # Inputs: sequence of shape (batch_size, seq_len, in_dim)
    # Output: sequence of shape (batch_size, seq_len, 1)
    # discriminator = LSTMDiscriminator(input_shape).to(args.device)
    discriminator = Discriminator(input_shape).to(args.device)

    loss_func = torch.nn.BCELoss()

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    for epoch in range(args.epoch):
        for idx, sigs_real in enumerate(train_loader):
            sigs_real = sigs_real.to(args.device)

            random_input = torch.randn(sigs_real.shape[0], args.latent_dim).to(args.device)
            sigs_fake = generator(random_input)

            real_label = torch.ones((sigs_real.size(0), 1), requires_grad=False).to(args.device)
            fake_label = torch.zeros((sigs_real.size(0), 1), requires_grad=False).to(args.device)

            #  Train Generator
            g_loss = loss_func(discriminator(sigs_fake), real_label)
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            #  Train Discriminator
            real_loss = loss_func(discriminator(sigs_real), real_label)
            fake_loss = loss_func(discriminator(sigs_fake.detach()), fake_label)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.zero_grad()
            optimizer_d.step()

            print((f"[Epoch {epoch} / {args.epoch}][Batch {idx} / {len(train_loader)}] "
                   f"[G loss: {g_loss.item()}] [D loss: {d_loss.item()}]"))
            batches_done = epoch * len(train_loader) + idx
            if batches_done % args.sample_interval == 0:
                np.save(f'gen_signal/np_{batches_done}.npy', sigs_fake.detach().numpy())


if __name__ == "__main__":
    args = parse_args()
    main()
