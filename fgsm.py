import torch
import torch.nn.functional as F

def fgsm_attack(model, images, labels, epsilon):
    images.requires_grad = True

    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)

    model.zero_grad()
    loss.backward()

    data_grad = images.grad.data

    perturbed_images = images + epsilon * data_grad.sign()
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images