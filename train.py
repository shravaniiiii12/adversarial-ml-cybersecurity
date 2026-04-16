import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model import SimpleCNN
from fgsm import fgsm_attack
from pgd import pgd_attack
from feature_squeeze import feature_squeeze

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

# dataset
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

# model
model = SimpleCNN().to(device)

# loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ================= TRAINING =================
model.train()

for epoch in range(1):
    for batch_idx, (images, labels) in enumerate(train_loader):

        images, labels = images.to(device), labels.to(device)

        # generate adversarial images
        adv_images = fgsm_attack(model, images, labels, epsilon=0.1)

        # combine clean + adversarial
        combined_images = torch.cat([images, adv_images])
        combined_labels = torch.cat([labels, labels])

        # forward
        outputs = model(combined_images)
        loss = criterion(outputs, combined_labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item()}")

print("Training Done")


# ================= EVALUATION =================
model.eval()

epsilons = [0, 0.05, 0.1, 0.2, 0.3]
accuracies = []

for eps in epsilons:
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # apply attack
        if eps != 0:
            adv_images = pgd_attack(model, images, labels, epsilon=eps)
        else:
            adv_images = images

        # apply defense
        adv_images = feature_squeeze(adv_images)

        outputs = model(adv_images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    accuracies.append(accuracy)
    print(f"Epsilon: {eps} | Accuracy: {accuracy:.2f}%")

# ================= GRAPH =================
plt.plot(epsilons, accuracies, marker='o')
plt.title("FGSM Attack vs Accuracy (With Adversarial Training)")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy (%)")
plt.grid()
plt.show()