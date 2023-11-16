import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

from functions import vq, vq_st

def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_size, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size, 5, 1, 0),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, z_dim * 2, 3, 1, 0),
            nn.BatchNorm2d(z_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, hidden_size, 3, 1, 0),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size, hidden_size, 5, 1, 0),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, x):
        # The encoder takes the input x and returns two tensors: mu and logvar
        # These tensors are chunked along dimension 1
        mu, logvar = self.encoder(x).chunk(2, dim=1)

        # q_z_x is the distribution of the encoded input
        # It is a Normal distribution with mean mu and standard deviation exp(logvar / 2)
        q_z_x = Normal(mu, logvar.mul(.5).exp())

        # p_z is the prior distribution
        # It is a Normal distribution with mean 0 and standard deviation 1
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))

        # kl_div is the KL divergence between the encoded distribution and the prior
        # It is calculated for each item in the batch and then averaged
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()

        # x_tilde is the decoded output
        # It is generated by sampling from the encoded distribution and passing the sample through the decoder
        x_tilde = self.decoder(q_z_x.rsample())

        # The forward function returns the decoded output and the KL divergence
        return x_tilde, kl_div
    

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size) # Hidden layer to hidden layer
        self.fc3 = nn.Linear(hidden_size, output_size) # Hidden layer to output layer
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the feature and spatial dimensions
        x = self.relu(self.fc1(x))  # Activation function after first layer
        x = self.relu(self.fc2(x))  # Activation function after second layer
        output = self.fc3(x)        # Output layer, no activation
        return output


class VAEPolicy(nn.Module):
    def __init__(self, input_dim, vae_hidden_size, z_dim, policy_hidden_size, policy_out_dim):
        super(VAEPolicy, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, vae_hidden_size, 4, 2, 1),
            nn.BatchNorm2d(vae_hidden_size),
            nn.ReLU(True),
            nn.Conv2d(vae_hidden_size, vae_hidden_size, 4, 2, 1),
            nn.BatchNorm2d(vae_hidden_size),
            nn.ReLU(True),
            nn.Conv2d(vae_hidden_size, vae_hidden_size, 5, 1, 0),
            nn.BatchNorm2d(vae_hidden_size),
            nn.ReLU(True),
            nn.Conv2d(vae_hidden_size, z_dim * 2, 3, 1, 0),
            nn.BatchNorm2d(z_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, vae_hidden_size, 3, 1, 0),
            nn.BatchNorm2d(vae_hidden_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(vae_hidden_size, vae_hidden_size, 5, 1, 0),
            nn.BatchNorm2d(vae_hidden_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(vae_hidden_size, vae_hidden_size, 4, 2, 1),
            nn.BatchNorm2d(vae_hidden_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(vae_hidden_size, input_dim, 4, 2, 1),
            nn.Tanh()
        )
        
        self.policy = PolicyNetwork(input_size=z_dim, hidden_size=policy_hidden_size, output_size=policy_out_dim)

    def forward(self, x):
        # The encoder takes the input x and returns two tensors: mu and logvar
        # These tensors are chunked along dimension 1
        mu, logvar = self.encoder(x).chunk(2, dim=1)

        # q_z_x is the distribution of the encoded input
        # It is a Normal distribution with mean mu and standard deviation exp(logvar / 2)
        q_z_x = Normal(mu, logvar.mul(.5).exp())

        # p_z is the prior distribution
        # It is a Normal distribution with mean 0 and standard deviation 1
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))

        # kl_div is the KL divergence between the encoded distribution and the prior
        # It is calculated for each item in the batch and then averaged
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()

        # x_tilde is the decoded output
        # It is generated by sampling from the encoded distribution and passing the sample through the decoder        
        sample = q_z_x.rsample()
        
        x_tilde = self.decoder(sample)
        # Pass the encoded representation through the policy network
        policy_output = self.policy(sample)

        return x_tilde, kl_div, policy_output



class VQEmbedding(nn.Module):
    """
    Vector Quantization Embedding Layer
    This layer takes a tensor z_e_x and maps it to a discrete latent space.
    The latent space is represented by a dictionary of embeddings, self.embedding.
    """
    def __init__(self, K, D):
        """
        Initialize the layer.
        Args:
            K: the number of embeddings
            D: the dimensionality of each embedding
        """
        super().__init__()
        self.embedding = nn.Embedding(K, D)  # The embedding dictionary
        self.embedding.weight.data.uniform_(-1./K, 1./K)  # Initialize the weights uniformly

    def forward(self, z_e_x):
        """
        Map the input tensor to its nearest embedding.
        Args:
            z_e_x: the input tensor
        Returns:
            latents: the tensor of nearest embeddings
        """
        # z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()  # Rearrange the tensor dimensions
        latents = vq(z_e_x, self.embedding.weight)  # Map the input tensor to its nearest embedding
        return latents

    def straight_through(self, z_e_x):
        """
        Perform a straight-through estimator of the gradients.
        This function is used during backpropagation.
        Args:
            z_e_x: the input tensor
        Returns:
            z_q_x: the tensor of nearest embeddings
            z_q_x_bar: the tensor of nearest embeddings with respect to the original tensor
        """
        # z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()  # Rearrange the tensor dimensions
        z_q_x, indices = vq_st(z_e_x, self.embedding.weight.detach())  # Map the input tensor to its nearest embedding
        # z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()  # Rearrange the tensor dimensions back

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)  # Select the embeddings by the indices
        z_q_x_bar = z_q_x_bar_flatten.view_as(z_e_x)  # Reshape the tensor to the original shape
        # z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()  # Rearrange the tensor dimensions back

        return z_q_x, z_q_x_bar


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.codebook = VQEmbedding(K, dim)     # This is very important.

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.fc = nn.Linear(dim, K * dim)  # Fully connected layer to output K vectors
        self.K = K
        self.dim = dim

        self.apply(weights_init)

    def encode(self, x):
        # Pass the input through the encoder
        z_e_x = self.image_encoder(x)
        # Now we need to vector quantize
        # print(z_e_x.size()) # torch.Size([16, 256, 7, 7])
        # Apply global average pooling to get a single vector per image
        z_e_x = z_e_x.mean(dim=[2, 3])
        # print(z_e_x.size()) # torch.Size([16, 256])
        # Pass through fully connected layer to get K vectors per image
        z_e_x = self.fc(z_e_x).view(-1, self.K, self.dim)
        # Quantize each vector using the codebook
        latents = torch.stack([self.codebook(vec) for vec in z_e_x.split(1, dim=1)])
        return latents

    def decode(self, latents):
        # Convert the latents to embeddings using the codebook
        print(latents.size())
        z_q_x = self.codebook.embedding(latents.long())
        # Pass the embeddings through the decoder to generate the output
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        # Pass the input through the encoder
        z_e_x = self.image_encoder(x)
        # Apply global average pooling to get a single vector per image
        z_e_x = z_e_x.mean(dim=[2, 3])
        # print(z_e_x.size()) # torch.Size([16, 256])
        # Pass through fully connected layer to get K vectors per image
        z_e_x = self.fc(z_e_x).view(-1, self.K, self.dim)
        # Quantize the output of the encoder using the codebook
        # Also get the straight-through estimator for the gradients
        sss = [self.codebook.straight_through(vec) for vec in z_e_x.split(1, dim=1)]
        latents = torch.stack([s[0] for s in sss])
        gradients = torch.stack([s[1] for s in sss])

        # z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        # Pass the straight-through estimator through the decoder to generate the output
        x_tilde = self.decode(latents)
        return x_tilde, z_e_x, gradients


class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return F.tanh(x) * F.sigmoid(y)


class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual

        self.class_cond_embedding = nn.Embedding(
            n_classes, 2 * dim
        )

        kernel_shp = (kernel // 2 + 1, kernel)  # (ceil(n/2), n)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.horiz_resid = nn.Conv2d(dim, dim, 1)

        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    def forward(self, x_v, x_h, h):
        if self.mask_type == 'A':
            self.make_causal()

        h = self.class_cond_embedding(h)
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        out_v = self.gate(h_vert + h[:, :, None, None])

        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)

        out = self.gate(v2h + h_horiz + h[:, :, None, None])
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)

        return out_v, out_h


class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10):
        super().__init__()
        self.dim = dim

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(input_dim, dim)

        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()

        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes)
            )

        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )

        self.apply(weights_init)

    def forward(self, x, label):
        shp = x.size() + (-1, )
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, W, W)

        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)

        return self.output_conv(x_h)

    def generate(self, label, shape=(8, 8), batch_size=64):
        param = next(self.parameters())
        x = torch.zeros(
            (batch_size, *shape),
            dtype=torch.int64, device=param.device
        )

        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x, label)
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )
        return x
