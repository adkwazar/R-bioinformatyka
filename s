transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

train_dataset = PathMNIST(split='train', transform=transform, download=False)
test_dataset  = PathMNIST(split='test', transform=transform, download=False)
