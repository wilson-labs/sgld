def get_cifar10(root=None, augment=True):
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms

    _CIFAR_TRAIN_TRANSFORM = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    _CIFAR_TEST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = CIFAR10(root=root, train=True, download=True,
                         transform=_CIFAR_TRAIN_TRANSFORM if augment else _CIFAR_TEST_TRANSFORM)

    test_data = CIFAR10(root=root, train=False, download=True,
                        transform=_CIFAR_TEST_TRANSFORM)

    return train_data, test_data
