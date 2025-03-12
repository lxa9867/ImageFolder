import torch
import torch.nn.functional as F

def add_perturbation(z, z_q, z_channels, codebook_norm, codebook, alpha, beta, delta):
    # reshape z -> (batch, height * width, channel) and flatten
    z = torch.einsum('b c h w -> b h w c', z).contiguous()
    z_flattened = z.view(-1, z_channels)

    if codebook_norm:
        z = F.normalize(z, p=2, dim=-1)
        z_flattened = F.normalize(z_flattened, p=2, dim=-1)
        embedding = F.normalize(codebook.weight, p=2, dim=-1)
    else:
        embedding = codebook.weight

    d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
        torch.sum(embedding ** 2, dim=1) - 2 * \
        torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))

    _, min_encoding_indices = torch.topk(d, delta, dim=1, largest=False)
    random_prob = torch.rand(min_encoding_indices.shape[0], device=d.device)
    random_idx = torch.randint(0, delta, random_prob.shape, device=d.device)
    random_idx = torch.where(random_prob > alpha, 0, random_idx)
    min_encoding_indices = min_encoding_indices[torch.arange(min_encoding_indices.size(0)), random_idx]

    perturbed_z_q = codebook(min_encoding_indices).view(z.shape)
    if codebook_norm:
        perturbed_z_q = F.normalize(perturbed_z_q, p=2, dim=-1)
    perturbed_z_q = z + (perturbed_z_q - z).detach()
    perturbed_z_q = torch.einsum('b h w c -> b c h w', perturbed_z_q)

    mask = torch.arange(z.shape[0], device=perturbed_z_q.device) < int(z.shape[0] * beta)
    mask = mask[:, None, None, None]

    return torch.where(mask, perturbed_z_q, z_q)

