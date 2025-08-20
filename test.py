import torch


def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """
    Normalize the angle to the range [0, π].

    Parameters
    ----------
    angle : torch.Tensor
        The angle in radians.

    Returns
    -------
    normalized_angle : torch.Tensor
        The normalized angle in the range [0, π].
    """
    # Step 1: Normalize angle to [-π, π]
    angle = torch.remainder(angle + torch.pi, 2 * torch.pi) - torch.pi

    # Step 2: Convert negative angles to positive
    angle = torch.where(angle < 0, angle + 2 * torch.pi, angle)

    # Step 3: Limit angle to [0, π]
    angle = torch.where(angle > torch.pi, 2 * torch.pi - angle, angle)

    return angle


# Example usage
angles = torch.tensor([-2 * torch.pi, -torch.pi, -torch.pi / 2, 0, torch.pi / 2, torch.pi, 2 * torch.pi])
normalized_angles = normalize_angle(angles)
print("Original angles:", angles)
print("Normalized angles:", normalized_angles)