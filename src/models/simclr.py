
import torch
import torch.nn as nn
import torchvision.models as models

class SimCLR(nn.Module):
    """
    A Simple Framework for Contrastive Learning of Visual Representations (SimCLR).
    """
    def __init__(self, base_model, out_dim):
        """
        Initializes the SimCLR model.

        Args:
            base_model (str): The name of the base model to use (e.g., 'resnet50').
            out_dim (int): The output dimension of the projection head.
        """
        super(SimCLR, self).__init__()
        self.backbone = self._get_base_model(base_model)
        dim_mlp = self.backbone.fc.in_features

        # Add a projection head
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

    def _get_base_model(self, model_name):
        """
        Returns the base model.

        Args:
            model_name (str): The name of the model.

        Returns:
            torch.nn.Module: The base model.
        """
        try:
            model = models.__dict__[model_name](pretrained=False, num_classes=10)
        except KeyError:
            raise NotImplementedError(
                f"Invalid model name: {model_name}. "
                f"Choose from: {', '.join(models.__dict__.keys())}"
            )
        return model

    def forward(self, x1, x2):
        """
        Performs a forward pass through the model.

        Args:
            x1 (torch.Tensor): The first augmented view of the input data.
            x2 (torch.Tensor): The second augmented view of the input data.

        Returns:
            torch.Tensor, torch.Tensor: The output of the projection head for each view.
        """
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        return z1, z2

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy Loss.
    """
    def __init__(self, temperature, batch_size):
        """
        Initializes the NTXentLoss.

        Args:
            temperature (float): The temperature parameter.
            batch_size (int): The batch size.
        """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z1, z2):
        """
        Performs a forward pass through the loss function.

        Args:
            z1 (torch.Tensor): The output of the projection head for the first view.
            z2 (torch.Tensor): The output of the projection head for the second view.

        Returns:
            torch.Tensor: The NT-Xent loss.
        """
        representations = torch.cat([z1, z2], dim=0)
        sim_matrix = self.similarity_f(representations.unsqueeze(1), representations.unsqueeze(0))

        # Create positive and negative masks
        l_pos = torch.diag(sim_matrix, self.batch_size)
        r_pos = torch.diag(sim_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = sim_matrix[~torch.eye(2 * self.batch_size, dtype=bool)].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(logits.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)

if __name__ == '__main__':
    # Example usage
    # You would typically get these from a data loader with augmentations
    dummy_x1 = torch.randn(4, 3, 224, 224)
    dummy_x2 = torch.randn(4, 3, 224, 224)

    # Initialize the model
    simclr_model = SimCLR(base_model='resnet18', out_dim=128)

    # Get the projections
    z1, z2 = simclr_model(dummy_x1, dummy_x2)

    # Initialize the loss function
    loss_fn = NTXentLoss(temperature=0.5, batch_size=4)

    # Calculate the loss
    loss = loss_fn(z1, z2)
    print(f"SimCLR NT-Xent Loss: {loss.item()}")
