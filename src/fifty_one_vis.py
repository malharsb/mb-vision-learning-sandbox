import fiftyone as fo
import fiftyone.zoo as foz
import matplotlib.pyplot as plt

# Download small but popular dataset (CIFAR-10)
dataset = foz.load_zoo_dataset("cifar10", split="test")
print(f"Dataset: {dataset.name}")
print(f"Total Samples: {len(dataset)}")

# Sampling
samples = dataset.take(5)
fig, axes = plt.subplots(1, 5, figsize=(15,5))

for ax, sample in zip(axes, samples):
    img = sample.filepath
    image = plt.imread(img)
    ax.imshow(image)
    ax.axis("off")
plt.show()

# Launch FiftyOne App (optional)
session = fo.launch_app(dataset)
session.wait()

