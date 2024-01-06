import torch
# Check if CUDA is available, and if so, use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def real_grad(state, data):
    mu = state[0]
    sigma = state[1]

    N = data.shape[0]

    m1 = torch.sum(data - mu)
    m2 = torch.sum((data - mu) ** 2)

    grad_mu = m1 / (sigma ** 2)
    grad_sigma = -N / sigma + m2 / (sigma ** 3)

    # Move gradients to GPU
    grad_mu = grad_mu.to(device)
    grad_sigma = grad_sigma.to(device)

    return torch.tensor([grad_mu, grad_sigma], dtype=torch.float32)
