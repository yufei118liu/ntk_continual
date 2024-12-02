import torch
import torch.nn as nn
import torch.distributions as dist
import torch.optim as optim

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior={'w_mean': 0.0, 'w_std': 1.0, 'b_mean': 0.0, 'b_std': 1.0}):
        super(BayesianLinear, self).__init__()

        self.prior_weight = dist.Normal(prior['w_mean'], prior['w_std'])
        self.prior_bias = dist.Normal(prior['b_mean'], prior['b_std'])
        self.weight_mu = nn.Parameter(torch.full((out_features, in_features), prior['w_mean']))
        self.weight_log_var = nn.Parameter(torch.full((out_features, in_features), prior['w_std']))
        self.bias_mu = nn.Parameter(torch.full((out_features,), prior['b_mean']))
        self.bias_log_var = nn.Parameter(torch.full((out_features,), prior['b_std']))
        
    def forward(self, x):
        # Sample weights and biases
        weight_sigma = torch.exp(0.5 * self.weight_log_var)
        bias_sigma = torch.exp(0.5 * self.bias_log_var)

        weight_distribution = dist.Normal(self.weight_mu, weight_sigma)
        bias_distribution = dist.Normal(self.bias_mu, bias_sigma)

        weights = weight_distribution.rsample()
        biases = bias_distribution.rsample()

        # Compute output
        return torch.matmul(x, weights.t()) + biases

    def kl_divergence(self):
        # Compute KL divergence between posterior and prior
        weight_sigma = torch.exp(0.5 * self.weight_log_var)
        bias_sigma = torch.exp(0.5 * self.bias_log_var)

        posterior_weight = dist.Normal(self.weight_mu, weight_sigma)
        posterior_bias = dist.Normal(self.bias_mu, bias_sigma)

        kl_div_weights = dist.kl_divergence(posterior_weight, self.prior_weight).sum()
        kl_div_biases = dist.kl_divergence(posterior_bias, self.prior_bias).sum()

        return kl_div_weights + kl_div_biases


class BNNMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BNNMLP, self).__init__()
        self.layer1 = BayesianLinear(input_dim, hidden_dim)
        self.layer2 = BayesianLinear(hidden_dim, hidden_dim)
        self.output_layer = BayesianLinear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

    def kl_divergence(self):
        # Sum KL divergence for all layers
        return self.layer1.kl_divergence() + self.layer2.kl_divergence() + self.output_layer.kl_divergence()

if __name__ == '__main__':
    # Test the model
    x_train = torch.randn(100, 10)  # 100 samples, 10 features
    y_train = torch.randint(0, 2, (100,))  # Binary classification (labels: 0 or 1)

    # Hyperparameters
    input_dim = 10
    hidden_dim = 50
    output_dim = 2
    num_epochs = 50
    batch_size = 16
    learning_rate = 1e-3
    beta = 1.0  # KL divergence scaling factor

    # Model, optimizer, and loss function
    bnn = BNNMLP(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(bnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        bnn.train()
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Forward pass
            outputs = bnn(x_batch)
            nll_loss = criterion(outputs, y_batch)
            kl_div = bnn.kl_divergence()

            # Total loss
            loss = nll_loss + beta * kl_div

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Training complete!")

