import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from torch.distributions.normal import Normal

from modules import VectorQuantizedVAE, VAE, VAEPolicy, to_scalar
from datasets import MiniImagenet

from tensorboardX import SummaryWriter

def train(data_loader, model, optimizer, args, writer):
    for images, labels in data_loader:
        images = images.to(args.device)
        labels = labels.to(args.device)

        optimizer.zero_grad()

        if args.arch == 'vqvae':
            x_tilde, z_e_x, z_q_x = model(images)
            # Reconstruction loss
            loss_recons = F.mse_loss(x_tilde, images)
            # Vector quantization objective
            loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
            # Commitment objective
            loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
            loss = loss_recons + loss_vq + args.beta * loss_commit
            # Logs
            writer.add_scalar('loss/train/reconstruction', loss_recons.item(), args.steps)
            writer.add_scalar('loss/train/quantization', loss_vq.item(), args.steps)
        
        elif args.arch == 'vae':
            x_tilde, kl_d = model(images)
            loss_recons = F.mse_loss(x_tilde, images, reduction='sum') / images.size(0)
            loss = loss_recons + kl_d
            nll = -Normal(x_tilde, torch.ones_like(x_tilde)).log_prob(images)
            log_px = nll.mean().item() - np.log(128) + kl_d.item()
            log_px /= np.log(2)
            # Logs
            writer.add_scalar('loss/train/reconstruction', loss_recons.item(), args.steps)
            writer.add_scalar('loss/train/kl-divergence', kl_d, args.steps)
            writer.add_scalar('loss/train/total', loss.item(), args.steps)

        elif args.arch == 'vaepolicy':
            x_tilde, kl_d, policy_output = model(images)
            loss_recons = F.mse_loss(x_tilde, images, reduction='sum') / images.size(0)
            # Calculate additional loss term for correct image label
            # labels_onehot = F.one_hot(labels, num_classes=args.num_actions)
            loss_policy = F.cross_entropy(policy_output, labels)
            loss = loss_recons + kl_d + args.policy_loss_coeff*loss_policy
            nll = -Normal(x_tilde, torch.ones_like(x_tilde)).log_prob(images)
            log_px = nll.mean().item() - np.log(128) + kl_d.item()
            log_px /= np.log(2)
            # Logs
            writer.add_scalar('loss/train/reconstruction', loss_recons.item(), args.steps)
            writer.add_scalar('loss/train/kl-divergence', kl_d, args.steps)
            writer.add_scalar('loss/train/policy', loss_policy.item(), args.steps)
            writer.add_scalar('loss/train/total', loss.item(), args.steps)

        loss.backward()
        optimizer.step()
        args.steps += 1

def test(data_loader, model, args, writer):
    if args.arch == 'vqvae':
        with torch.no_grad():
            loss_recons, loss_vq = 0., 0.
            for images, _ in data_loader:
                images = images.to(args.device)
                x_tilde, z_e_x, z_q_x = model(images)
                loss_recons += F.mse_loss(x_tilde, images)
                loss_vq += F.mse_loss(z_q_x, z_e_x)

            loss_recons /= len(data_loader)
            loss_vq /= len(data_loader)

        # Logs
        writer.add_scalar('loss/test/reconstruction', loss_recons.item(), args.steps)
        writer.add_scalar('loss/test/quantization', loss_vq.item(), args.steps)

        return loss_recons.item(), loss_vq.item()
    elif args.arch == 'vae':
        with torch.no_grad():
            loss_recons = 0.
            for images, _ in data_loader:
                images = images.to(args.device)
                x_tilde, kl_d = model(images)
                loss_recons = F.mse_loss(x_tilde, images, reduction='sum') / images.size(0)
                loss_with_kl = loss_recons + kl_d

            loss_recons /= len(data_loader)
            loss_with_kl /= len(data_loader)

        # Logs
        writer.add_scalar('loss/test/reconstruction', loss_recons.item(), args.steps)
        writer.add_scalar('loss/test/kl-divergence', kl_d, args.steps)
        writer.add_scalar('loss/test/total', loss_with_kl.item(), args.steps)

        return loss_recons.item(), loss_with_kl.item()
    elif args.arch == 'vaepolicy':
        with torch.no_grad():
            loss_recons = 0.
            for images, labels in data_loader:
                images = images.to(args.device)
                labels = labels.to(args.device)
                x_tilde, kl_d, policy_output = model(images)
                loss_policy = F.cross_entropy(policy_output, labels)
                loss_recons = F.mse_loss(x_tilde, images, reduction='sum') / images.size(0)
                loss_with_kl_and_policy = loss_recons + kl_d + args.policy_loss_coeff*loss_policy

            loss_recons /= len(data_loader)
            loss_with_kl_and_policy /= len(data_loader)

        # Logs
        writer.add_scalar('loss/test/reconstruction', loss_recons.item(), args.steps)
        writer.add_scalar('loss/test/kl-divergence', kl_d, args.steps)
        writer.add_scalar('loss/test/policy', loss_policy.item(), args.steps)
        writer.add_scalar('loss/test/total', loss_with_kl_and_policy.item(), args.steps)

        return loss_recons.item(), loss_with_kl_and_policy.item()

def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        if args.arch == 'vae':
            x_tilde, _ = model(images)
        elif args.arch == 'vaepolicy':
            x_tilde, _, _ = model(images)
    return x_tilde

def main(args):
    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_filename = './models/{0}'.format(args.output_folder)

    if args.dataset in ['mnist', 'fashion-mnist', 'cifar10']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        if args.dataset == 'mnist':
            # Define the train & test datasets
            train_dataset = datasets.MNIST(args.data_folder, train=True,
                download=True, transform=transform)
            test_dataset = datasets.MNIST(args.data_folder, train=False,
                transform=transform)
            num_channels = 1
        elif args.dataset == 'fashion-mnist':
            # Define the train & test datasets
            train_dataset = datasets.FashionMNIST(args.data_folder,
                train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(args.data_folder,
                train=False, transform=transform)
            num_channels = 1
        elif args.dataset == 'cifar10':
            # Define the train & test datasets
            train_dataset = datasets.CIFAR10(args.data_folder,
                train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(args.data_folder,
                train=False, transform=transform)
            num_channels = 3
        valid_dataset = test_dataset
    elif args.dataset == 'miniimagenet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Define the train, valid & test datasets
        train_dataset = MiniImagenet(args.data_folder, train=True,
            download=True, transform=transform)
        valid_dataset = MiniImagenet(args.data_folder, valid=True,
            download=True, transform=transform)
        test_dataset = MiniImagenet(args.data_folder, test=True,
            download=True, transform=transform)
        num_channels = 3

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=16, shuffle=True)

    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(test_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, value_range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)

    # Instantiate model
    if args.arch == 'vae':
        model = VAE(input_dim=num_channels, hidden_size=args.vae_hidden_size, z_dim=args.latent_dim).to(args.device)
    elif args.arch == 'vqvae':
        model = VectorQuantizedVAE(num_channels, args.vae_hidden_size, args.latent_dim).to(args.device)
    elif args.arch == 'vaepolicy':
        model = VAEPolicy(input_dim=num_channels, vae_hidden_size=args.vae_hidden_size, z_dim=args.latent_dim, policy_hidden_size=args.latent_dim, policy_out_dim=args.num_actions).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Generate the samples first once
    reconstruction = generate_samples(fixed_images, model, args)
    grid = make_grid(reconstruction.cpu(), nrow=8, value_range=(-1, 1), normalize=True)
    writer.add_image('reconstruction', grid, 0)

    best_loss = -1.
    for epoch in range(args.num_epochs):
        train(train_loader, model, optimizer, args, writer)
        loss, _ = test(valid_loader, model, args, writer)

        reconstruction = generate_samples(fixed_images, model, args)
        grid = make_grid(reconstruction.cpu(), nrow=8, value_range=(-1, 1), normalize=True)
        writer.add_image('reconstruction', grid, epoch + 1)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                torch.save(model.state_dict(), f)
        with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
            torch.save(model.state_dict(), f)

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')

    # General
    parser.add_argument('--data-folder', type=str, default='/users/bspiegel/data/bspiegel/mnist',
        help='name of the data folder')
    parser.add_argument('--dataset', type=str, default='mnist',
        help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet)')
    
    # Architecture
    parser.add_argument('--arch', type=str, default='vae',
        help='which architecture to use')

    # Latent space
    parser.add_argument('--latent-dim', type=int, default=128,
        help='size of the latent vectors (default: 256)')
    parser.add_argument('--vae-hidden-size', type=int, default=256,
        help='size of the hidden layers in the vae (default: 256)')
    # parser.add_argument('--k', type=int, default=128,
    #     help='number of latent vectors (default: 128)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=128,
        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=25,
        help='number of epochs (default: 25)')
    parser.add_argument('--lr', type=float, default=1e-3,
        help='learning rate for Adam optimizer (default: 1e-3 good for vae) (2e-4 was good for vqvae)')
    parser.add_argument('--beta', type=float, default=1.0,
        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')
    parser.add_argument('--policy-loss-coeff', type=float, default=1.0,
        help='coefficient for policy loss (default: 1.0)')
    
    # RL experiment params
    parser.add_argument('--num-actions', type=int, default=10,
        help='number of actions for the policy ouput (default: 10)')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='vqvae',
        help='name of the output folder (default: vqvae)')
    parser.add_argument('--num-workers', type=int, default=4,
        help='number of workers for trajectories sampling (default: 4)')
    parser.add_argument('--device', type=str, default='cuda',
        help='set the device (cpu or cuda, default: cuda)')
    parser.add_argument('--verbose', type=bool, default=False,
        help='print statements after each epoch')

    args = parser.parse_args()

    # Create logs and models folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./models'):
        os.makedirs('./models')
    # Device
    if args.device == 'cuda':
        args.device = torch.device('cuda'
            if torch.cuda.is_available() else 'cpu')

    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])
    if not os.path.exists('./models/{0}'.format(args.output_folder)):
        os.makedirs('./models/{0}'.format(args.output_folder))
    args.steps = 0

    main(args)
