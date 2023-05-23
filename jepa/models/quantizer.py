from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

def l2_normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Function to apply L2 normalization along a specific dimension in a tensor.

    Parameters:
    -----------
    tensor : torch.Tensor
        The tensor to be normalized.
    dim : int
        The dimension along which L2 normalization is to be applied.

    Returns:
    --------
    torch.Tensor
        The L2 normalized tensor.
    """
    return F.normalize(tensor, p=2, dim=dim)

def entropy_loss(affinity, loss_type="softmax", temperature=1.0):
    """Calculates the entropy loss."""
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = torch.nn.functional.softmax(flat_affinity, dim=-1)
    log_probs = torch.nn.functional.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    elif loss_type == "argmax":
        codes = torch.argmax(flat_affinity, dim=-1)
        onehots = torch.zeros_like(flat_affinity)
        onehots.scatter_(1, codes.unsqueeze(1), 1)
        onehots = probs - (probs - onehots).detach()
        target_probs = onehots
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss

class ExponentialMovingAverage:
    def __init__(self, tensor, decay):
        self.decay = decay
        self.shadow = tensor.clone()

    def update(self, tensor):
        self.shadow.mul_(self.decay).add_(tensor, alpha=1 - self.decay)

    def __call__(self, tensor):
        self.update(tensor)

    def apply(self, parameters):
        parameters.data.copy_(self.shadow)

    def reset(self):
        self.shadow.zero_()

    def get(self):
        return self.shadow

def _kpoints(data, k):
    """
    Randomly pick k points in the dataset as initial centroids.

    Args:
        data (torch.Tensor): The data from which to select points, of shape (N, dim).
        k (int): The number of points to select.

    Returns:
        kpoints (torch.Tensor): The selected points, of shape (k, dim).
    """
    device = data.device
    rand_indices = torch.randperm(data.size(0), device=device)[:k]
    kpoints = data[rand_indices]

    if dist.is_initialized():
        # Gather the selected points from all devices to a list
        gathered_kpoints = [torch.zeros_like(kpoints, device=device) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_kpoints, kpoints)

        # Average the selected points across all devices
        kpoints = torch.mean(torch.stack(gathered_kpoints), dim=0)

    return kpoints

def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    """
    Perform K-means clustering on input samples.

    Args:
        samples (torch.Tensor): The data to be clustered, of shape (N, dim), where N is the number of data points
                                and dim is the dimension of each data point.
        num_clusters (int): The number of clusters to form.
        num_iters (int, optional): The number of iterations for the K-means algorithm. Default is 10.
        use_cosine_sim (bool, optional): If True, use cosine similarity instead of Euclidean distance. 
                                          Default is False.

    Returns:
        means (torch.Tensor): The final cluster centroids, of shape (num_clusters, dim).
        bins (torch.Tensor): The number of samples in each cluster, of shape (num_clusters).
    """
    dim, dtype = samples.shape[-1], samples.dtype
    means = _kpoints(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
            diffs = samples.unsqueeze(1) - means.unsqueeze(0)
            dists = -(diffs ** 2).sum(dim=-1)
        else:
            dists = -torch.cdist(samples, means, p=2)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.clone().detach().masked_fill(zero_mask, 1)

        if dist.is_initialized():
            # All-reduce the bins tensor across all GPUs
            dist.all_reduce(bins)
            dist.all_reduce(bins_min_clamped)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, buckets[:, None].repeat(1, dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2_normalize(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

        if dist.is_initialized():
            # All-reduce the new_means tensor across all GPUs
            dist.all_reduce(means)

    return means, bins

class VectorQuantizer(nn.Module):
    """Basic vector quantizer."""
    def __init__(self, config):
        super(VectorQuantizer, self).__init__()

        self.config = config

        # Define codebook as Parameter that will be learned
        self.codebook = nn.Parameter(
            torch.empty(self.config.vqvae.codebook_size, self.config.vqvae.embed_dim))
        self.cluster_size = torch.zeros(self.config.vqvae.codebook_size)
        
        self.codebook_ema = ExponentialMovingAverage(self.codebook.data, self.config.vqvae.ema_decay)
        self.cluster_size_ema = ExponentialMovingAverage(self.cluster_size, self.config.vqvae.ema_decay)
        self.kmeans_initialized = False

        nn.init.kaiming_uniform_(self.codebook, mode='fan_in', nonlinearity='relu')

    def initialize_codebook(self, x):
        if self.kmeans_initialized:
            return
        
        # Perform K-means clustering to initialize the codebook
        with torch.no_grad():
            codebook, cluster_size = kmeans(
                x, 
                self.config.vqvae.codebook_size, 
                num_iters=self.config.vqvae.kmeans_iters)
            self.codebook.data.copy_(codebook)
            self.cluster_size.copy_(cluster_size)
            self.codebook_ema.apply(self.codebook.data)
            self.cluster_size_ema.apply(self.cluster_size)
            self.kmeans_initialized = True

    def forward(self, x):
        x_shape = x.shape
        flat_x = x.view(-1, x_shape[-1])
        if self.config.vqvae.l2_normalize:
            flat_x = l2_normalize(flat_x, dim=-1)

        self.initialize_codebook(flat_x)
        
        # Calculate distances and encodings
        if not self.config.vqvae.l2_normalize:
            distances = torch.cdist(flat_x, self.codebook, p=2)
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        elif self.config.vqvae.l2_normalize:
            distances = torch.einsum('b d, n d -> b n', flat_x, l2_normalize(self.codebook, dim=-1)) # 'n d -> d n'
            encoding_indices = torch.argmax(distances, dim=1).unsqueeze(1)
        
        encodings = F.one_hot(encoding_indices, self.codebook.shape(0)).type(x.dtype)
        quantized = self.quantize(encodings)

        if self.train:
            e_latent_loss = torch.mean((quantized.detach() - x) ** 2) * self.config.vqvae.commitment_cost
            q_latent_loss = torch.mean((quantized - x.detach()) ** 2)

            entropy_loss = 0.0
            if self.config.vqvae.entropy_loss_ratio != 0:
                entropy_loss = entropy_loss(-distances,
                                            loss_type=self.config.vqvae.entropy_loss_type,
                                            temperature=self.config.vqvae.entropy_temperature
                                        ) * self.config.vqvae.entropy_loss_ratio

            loss = e_latent_loss + q_latent_loss + entropy_loss

            self.cluster_size_ema(encodings.sum(0))
            self.codebook_ema(torch.matmul(flat_x.t(), encodings))

            n = self.cluster_size_ema.get().sum()
            smoothed_cluster_size = ((self.cluster_size_ema.get() + self.config.vqvae.eps) /
                                    (n + self.config.vqvae.codebook_size * self.config.vqvae.eps) * n)
            normalized_codebook = self.codebook_ema.get() / smoothed_cluster_size.unsqueeze(1)

            if self.config.vqvae.l2_normalize:
                normalized_codebook = l2_normalize(normalized_codebook, dim=-1)

            self.codebook.data.copy_(normalized_codebook)

            result_dict = {
                "quantizer_loss": loss,
                "e_latent_loss": e_latent_loss,
                "q_latent_loss": q_latent_loss,
                "entropy_loss": entropy_loss,
            }

            quantized = x + (quantized - x).detach()
        else:
            result_dict = {}

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        result_dict.update({
            "encodings": encodings,
            "encoding_indices": encoding_indices,
            "raw": x,
            "perplexity": perplexity,
        })

        return quantized, result_dict

    def quantize(self, encodings):
        return torch.matmul(encodings, self.codebook)

    def decode_ids(self, ids):
        return self.codebook[ids]
