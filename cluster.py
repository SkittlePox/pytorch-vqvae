import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from torch.distributions.normal import Normal

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from modules import VectorQuantizedVAE, VAE, to_scalar
from datasets import MiniImagenet

from tensorboardX import SummaryWriter

def cluster(mu, iteration, args, writer):
    # Reshape the data for clustering (flattening each tensor)
    data_for_clustering = mu.reshape(mu.shape[0], -1)

    # K-means clustering
    n_clusters = args.num_clusters  # You can choose the number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    kmeans_labels = kmeans.fit_predict(data_for_clustering)

    # GMM clustering
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    gmm_labels = gmm.fit_predict(data_for_clustering)

    # Visualization (if needed)
    # This is an example of how you might visualize the results.
    # Adjust the visualization code according to your specific needs.
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(data_for_clustering[:, 0], data_for_clustering[:, 1], c=kmeans_labels)
    plt.title('K-means Clustering')

    plt.subplot(1, 2, 2)
    plt.scatter(data_for_clustering[:, 0], data_for_clustering[:, 1], c=gmm_labels)
    plt.title('GMM Clustering')

    if not os.path.exists(args.model_folder+"/cluster-figs"):
        os.makedirs(args.model_folder+"/cluster-figs")

    plt.savefig(args.model_folder+f"/cluster-figs/fig_{iteration}.png")
    plt.close()

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
        writer.add_scalar('loss/test/total', loss_with_kl.item(), args.steps)

        return loss_recons.item(), loss_with_kl.item()

def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _ = model(images)
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

    # # Fixed images for Tensorboard
    # fixed_images, _ = next(iter(test_loader))
    # fixed_grid = make_grid(fixed_images, nrow=8, value_range=(-1, 1), normalize=True)
    # writer.add_image('original', fixed_grid, 0)

    # Instantiate model
    if args.arch == 'vae':
        model = VAE(input_dim=num_channels, dim=args.hidden_size*2, z_dim=args.hidden_size).to(args.device)
    elif args.arch == 'vqvae':
        model = VectorQuantizedVAE(num_channels, args.hidden_size, args.k).to(args.device)

    for i in range(1, args.num_models + 1):
        print(f"Clustering iteration {i}")
        model_path = os.path.join(args.model_folder, f'model_{i}.pt')
        model.load_state_dict(torch.load(model_path))
        if args.arch == 'vae':
            with torch.no_grad():
                encodings = []
                for images, _ in train_loader:
                    images = images.to(args.device)
                    mu, logvar = model.encoder(images).chunk(2, dim=1)
                    # I will just take the mean params, it's deterministic. Eventually I may want to sample.
                    mu_detached = mu.detach().cpu()
                    encodings.append(mu_detached)

                encodings = torch.cat(encodings, dim=0)
                encodings = encodings[:args.num_datapoints]
                cluster(encodings, i, args, writer)

            # Logs
            # writer.add_scalar('loss/test/reconstruction', loss_recons.item(), args.steps)
            # writer.add_scalar('loss/test/total', loss_with_kl.item(), args.steps)

    return
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

    parser = argparse.ArgumentParser(description='Clustering')

    # General
    parser.add_argument('--data-folder', type=str, default='/users/bspiegel/data/bspiegel/mnist',
        help='name of the data folder')
    parser.add_argument('--dataset', type=str, default='mnist',
        help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet)')
    parser.add_argument('--model-folder', type=str, default='models/vae-k128-12070364',
        help='the folder containing pytorch models')
    parser.add_argument('--num-models', type=int, default=25,
        help='number of models in the folder')

    
    # Architecture
    parser.add_argument('--arch', type=str, default='vae',
        help='which architecture to use')

    # Latent space
    parser.add_argument('--hidden-size', type=int, default=128,
        help='size of the latent vectors (default: 128)')
    parser.add_argument('--k', type=int, default=128,
        help='number of latent vectors (default: 128)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=128,
        help='batch size (default: 128)')
    # parser.add_argument('--num-epochs', type=int, default=25,
    #     help='number of epochs (default: 25)')
    # parser.add_argument('--lr', type=float, default=1e-3,
    #     help='learning rate for Adam optimizer (default: 1e-3 good for vae) (2e-4 was good for vqvae)')
    # parser.add_argument('--beta', type=float, default=1.0,
    #     help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    # Visualization
    parser.add_argument('--num-clusters', type=int, default=10,
        help='number of clusters')
    parser.add_argument('--num-datapoints', type=int, default=1000,
        help='number of datapoints to cluster')
    
    
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
