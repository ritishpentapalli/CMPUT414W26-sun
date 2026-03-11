import io
import matplotlib.pyplot as plt
from PIL import Image

def get_bpp(file_size_bytes, width, height):
    """Calculates bits-per-pixel based on actual file size."""
    return (file_size_bytes * 8) / (width * height)

def compress_jp2(img, ratio):
    """Compresses to JP2 using a specific ratio and returns image + actual bpp."""
    buf = io.BytesIO()
    # quality_layers defines the compression ratio (e.g., 20 means 20:1)
    img.save(buf, format="JPEG2000", quality_layers=[ratio])
    size = buf.tell()
    buf.seek(0)
    return Image.open(buf), get_bpp(size, img.width, img.height)

# 1. Load your original image (Must be RGB for these ratios to work)
try:
    original = Image.open("lena_gray.gif").convert("RGB")
except FileNotFoundError:
    print("Please ensure 'original.png' is in the current directory.")
    exit()

# 2. Perform Compressions
# Target 0.6 bpp -> 24/0.6 = 40 ratio
img_med, bpp_med = compress_jp2(original, 40)

# Target 0.015 bpp -> 24/0.015 = 1600 ratio
img_low, bpp_low = compress_jp2(original, 350)

# 3. Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

data = [
    (original, "Original Image", 24.0),
    (img_med, "Medium Bitrate", bpp_med),
    (img_low, "Low Bitrate", bpp_low)
]

for ax, (img, label, bpp) in zip(axes, data):
    ax.imshow(img)
    ax.set_title(f"{label}\n{bpp:.4f} BPP", fontsize=14)
    ax.axis("off")

plt.tight_layout()
plt.show()