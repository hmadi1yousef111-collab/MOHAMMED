import numpy as np
image = np.array([
    [0, 50, 100],
    [150, 200, 250],
    [255, 128, 64]
])

print(image)
print("Shape:", image.shape)
print("Data type:", image.dtype)
print("Minimum:", image.min())
print("Maximum:", image.max())
print("Mean:", image.mean())
