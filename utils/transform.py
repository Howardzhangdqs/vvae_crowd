import torch
from torchvision import transforms


def transform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    return transforms.Compose([
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize(mean, std)
    ])


def inverse_transform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    return transforms.Compose([
        transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std]),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
        transforms.Lambda(lambda x: (x * 255).byte())
    ])


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import PIL.Image as Image

    img = np.random.randint(0, 256, (256, 256, 3))
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img).float()

    transform_fn = transform()
    inverse_transform_fn = inverse_transform()

    img_transformed = transform_fn(img)
    img_inverse_transformed = inverse_transform_fn(img_transformed)

    img = img.numpy().transpose(1, 2, 0)
    img_transformed = img_transformed.numpy().transpose(1, 2, 0)
    img_inverse_transformed = img_inverse_transformed.numpy().transpose(1, 2, 0)
    
    # print the min and max values of the images
    print(img.min(), img.max())
    print(img_transformed.min(), img_transformed.max())
    print(img_inverse_transformed.min(), img_inverse_transformed.max())

    plt.subplot(131)
    plt.imshow(img.astype(np.uint8))
    plt.title("Original")
    plt.subplot(132)
    plt.imshow((img_transformed).astype(np.float32))
    plt.title("Transformed")
    plt.subplot(133)
    plt.imshow(img_inverse_transformed.astype(np.uint8))
    plt.title("Inverse Transformed")
    
    # save the transformed image
    plt.savefig("transform.png")
