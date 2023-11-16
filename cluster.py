import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from torch.distributions.normal import Normal

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

from modules import VectorQuantizedVAE, VAE, VAEPolicy, to_scalar
from datasets import MiniImagenet

from tensorboardX import SummaryWriter

def cluster(mu, labels, iteration, args, writer, dim=2):
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

    data_for_reduction = data_for_clustering

    pca = PCA(n_components=dim)
    pca_result = pca.fit_transform(data_for_reduction)

    # Apply t-SNE
    tsne = TSNE(n_components=dim, random_state=0)
    tsne_result = tsne.fit_transform(data_for_reduction)

    # Apply UMAP
    umap_result = umap.UMAP(n_components=dim).fit_transform(data_for_reduction)

    # Visualization
    if dim == 2:
        fig = plt.figure(figsize=(18, 18))
        
        ax1 = fig.add_subplot(331)
        ax1.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_labels)
        ax1.set_title(f'PCA Result with K-Means (k={n_clusters})')
        
        ax2 = fig.add_subplot(332)
        ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], c=kmeans_labels)
        ax2.set_title(f't-SNE Result with K-Means (k={n_clusters})')
        
        ax3 = fig.add_subplot(333)
        ax3.scatter(umap_result[:, 0], umap_result[:, 1], c=kmeans_labels)
        ax3.set_title(f'UMAP Result with K-Means (k={n_clusters})')
        
        ax4 = fig.add_subplot(334)
        ax4.scatter(pca_result[:, 0], pca_result[:, 1], c=gmm_labels)
        ax4.set_title(f'PCA Result with GMM (k={n_clusters})')
        
        ax5 = fig.add_subplot(335)
        ax5.scatter(tsne_result[:, 0], tsne_result[:, 1], c=gmm_labels)
        ax5.set_title(f't-SNE Result with GMM (k={n_clusters})')
        
        ax6 = fig.add_subplot(336)
        ax6.scatter(umap_result[:, 0], umap_result[:, 1], c=gmm_labels)
        ax6.set_title(f'UMAP Result with GMM (k={n_clusters})')
        
        ax7 = fig.add_subplot(337)
        ax7.scatter(pca_result[:, 0], pca_result[:, 1], c=labels)
        ax7.set_title(f'PCA Result with True Labels')
        
        ax8 = fig.add_subplot(338)
        ax8.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels)
        ax8.set_title(f't-SNE Result with True Labels')
        
        ax9 = fig.add_subplot(339)
        ax9.scatter(umap_result[:, 0], umap_result[:, 1], c=labels)
        ax9.set_title(f'UMAP Result with True Labels')
    
    elif dim == 3:
        fig = plt.figure(figsize=(18, 18))
        
        ax1 = fig.add_subplot(331, projection='3d')
        ax1.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=kmeans_labels)
        ax1.set_title(f'PCA Result with K-Means (k={n_clusters})')
        
        ax2 = fig.add_subplot(332, projection='3d')
        ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], tsne_result[:, 2], c=kmeans_labels)
        ax2.set_title(f't-SNE Result with K-Means (k={n_clusters})')
        
        ax3 = fig.add_subplot(333, projection='3d')
        ax3.scatter(umap_result[:, 0], umap_result[:, 1], umap_result[:, 2], c=kmeans_labels)
        ax3.set_title(f'UMAP Result with K-Means (k={n_clusters})')
        
        ax4 = fig.add_subplot(334, projection='3d')
        ax4.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=gmm_labels)
        ax4.set_title(f'PCA Result with GMM (k={n_clusters})')
        
        ax5 = fig.add_subplot(335, projection='3d')
        ax5.scatter(tsne_result[:, 0], tsne_result[:, 1], tsne_result[:, 2], c=gmm_labels)
        ax5.set_title(f't-SNE Result with GMM (k={n_clusters})')
        
        ax6 = fig.add_subplot(336, projection='3d')
        ax6.scatter(umap_result[:, 0], umap_result[:, 1], umap_result[:, 2], c=gmm_labels)
        ax6.set_title(f'UMAP Result with GMM (k={n_clusters})')
        
        ax7 = fig.add_subplot(337, projection='3d')
        ax7.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=labels)
        ax7.set_title(f'PCA Result with True Labels')
        
        ax8 = fig.add_subplot(338, projection='3d')
        ax8.scatter(tsne_result[:, 0], tsne_result[:, 1], tsne_result[:, 2], c=labels)
        ax8.set_title(f't-SNE Result with True Labels')
        
        ax9 = fig.add_subplot(339, projection='3d')
        ax9.scatter(umap_result[:, 0], umap_result[:, 1], umap_result[:, 2], c=labels)
        ax9.set_title(f'UMAP Result with True Labels')

    if not os.path.exists(args.model_folder+"/cluster-figs"):
        os.makedirs(args.model_folder+"/cluster-figs")

    plt.savefig(args.model_folder+f"/cluster-figs/fig_{iteration}_{dim}d.png")
    plt.close()

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
        model = VAE(input_dim=num_channels, hidden_size=args.vae_hidden_size, z_dim=args.latent_dim).to(args.device)
    elif args.arch == 'vqvae':
        model = VectorQuantizedVAE(num_channels, args.vae_hidden_size, args.latent_dim).to(args.device)
    elif args.arch == 'vaepolicy':
        model = VAEPolicy(input_dim=num_channels, vae_hidden_size=args.vae_hidden_size, z_dim=args.latent_dim, policy_hidden_size=args.latent_dim, policy_out_dim=args.num_actions).to(args.device)

    for i in range(1, args.num_models + 1):
        print(f"Clustering model {i}")
        model_path = os.path.join(args.model_folder, f'model_{i}.pt')
        model.load_state_dict(torch.load(model_path))
        if args.arch in ('vae', 'vaepolicy'):
            with torch.no_grad():
                encodings = []
                labels = []
                for images, labels_ in train_loader:
                    images = images.to(args.device)
                    mu, logvar = model.encoder(images).chunk(2, dim=1)
                    # I will just take the mean params, it's deterministic. Eventually I may want to sample.
                    mu_detached = mu.detach().cpu()
                    encodings.append(mu_detached)
                    labels.append(labels_)

                    if len(encodings) * args.batch_size > args.num_datapoints:
                        break

                encodings = torch.cat(encodings, dim=0)
                encodings = encodings[:args.num_datapoints]
                labels = torch.cat(labels, dim=0)
                labels = labels[:args.num_datapoints]
                
                cluster(encodings, labels, i, args, writer, dim=2)

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
    parser.add_argument('--log-folder', type=str, default='logs/vae-k128-12070364',
        help='the folder containing tensorboard logs')
    parser.add_argument('--num-models', type=int, default=25,
        help='number of models in the folder')

    
    # Architecture
    parser.add_argument('--arch', type=str, default='vae',
        help='which architecture to use')

    # Latent space
    parser.add_argument('--latent-dim', type=int, default=128,
        help='size of the latent vectors (default: 256)')
    parser.add_argument('--vae-hidden-size', type=int, default=256,
        help='size of the hidden layers in the vae (default: 256)')

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
