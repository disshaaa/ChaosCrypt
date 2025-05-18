import cv2
import numpy as np
import os
from chaos_aes_gui import encrypt_file, decrypt_file

def calculate_npcr_uaci(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("Images must be the same size")
    diff = img1 != img2
    npcr = np.sum(diff) / diff.size * 100
    uaci = np.mean(np.abs(img1.astype(np.int16) - img2.astype(np.int16)) / 255) * 100
    return npcr, uaci

# Step 1: Load original image
img_original = cv2.imread("sampleimage.png")
img_modified = img_original.copy()
img_modified[0, 0, 0] = (img_modified[0, 0, 0] + 1) % 256

# Save both original and modified copies
cv2.imwrite("img1.png", img_original)
cv2.imwrite("img2.png", img_modified)

# Step 2: Encrypt both (they will delete original files, so use copies)
encrypt_file("img1.png", r=3.9, seed=0.5, aes_key="1234567890abcdef")  # creates img1.png.enc
encrypt_file("img2.png", r=3.9, seed=0.5, aes_key="1234567890abcdef")  # creates img2.png.enc

# Step 3: Decrypt and rename output to avoid overwriting
decrypt_file("img1.png.enc", r=3.9, seed=0.5, aes_key="1234567890abcdef")  # recreates img1.png
os.replace("img1.png", "dec_img1.png")  # rename after decryption

decrypt_file("img2.png.enc", r=3.9, seed=0.5, aes_key="1234567890abcdef")  # recreates img2.png
os.replace("img2.png", "dec_img2.png")  # rename after decryption

# Step 4: Compare decrypted images
with open("img1.png.enc", "rb") as f:
    enc1 = np.frombuffer(f.read(), dtype=np.uint8)
with open("img2.png.enc", "rb") as f:
    enc2 = np.frombuffer(f.read(), dtype=np.uint8)

# Ensure they are same size
min_len = min(len(enc1), len(enc2))
enc1 = enc1[:min_len]
enc2 = enc2[:min_len]

# NPCR and UACI for byte-level data
diff = enc1 != enc2
npcr = np.sum(diff) / diff.size * 100
uaci = np.mean(np.abs(enc1.astype(np.int16) - enc2.astype(np.int16)) / 255) * 100

print(f"NPCR (on ciphertext): {npcr:.2f}%")
print(f"UACI (on ciphertext): {uaci:.2f}%")
