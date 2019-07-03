from mnist import MNIST
from PIL import Image, ImageDraw

# Loading the dataset dataset
mndata = MNIST('./data/')
images, labels = mndata.load_training()

# Picking an image from the dataset. Can use a RNG here but picking manually
i = 4
image, label = images[i], labels[i]

# Print the image
output = Image.new("L", (28, 28))
output.putdata(image)
output.save("output.png")

# Print label
print(label)
