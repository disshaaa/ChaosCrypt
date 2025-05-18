import os
import zlib
import math
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes
from docx import Document
from PyPDF2 import PdfReader

def read_file(filepath):
    if filepath.endswith(".txt"):
        with open(filepath, "rb") as f:
            return f.read()
    elif filepath.endswith(".docx"):
        doc = Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs]).encode()
    elif filepath.endswith(".pdf"):
        reader = PdfReader(filepath)
        return "\n".join([page.extract_text() or "" for page in reader.pages]).encode()
    else:
        raise ValueError("Unsupported file type")

def encrypt_aes(data, key):
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(data, AES.block_size))
    return cipher.iv + ct_bytes  # prepend IV for uniqueness

def byte_diff_percentage(data1, data2):
    min_len = min(len(data1), len(data2))
    diffs = sum(b1 != b2 for b1, b2 in zip(data1[:min_len], data2[:min_len]))
    return (diffs / min_len) * 100

def calculate_entropy(data):
    if not data:
        return 0
    prob = [float(data.count(byte)) / len(data) for byte in set(data)]
    return -sum(p * math.log2(p) for p in prob)

def compression_ratio(data):
    compressed = zlib.compress(data)
    return len(compressed) / len(data) if len(data) > 0 else 0

def avalanche_effect(cipher1, cipher2):
    bits1 = ''.join(f'{b:08b}' for b in cipher1)
    bits2 = ''.join(f'{b:08b}' for b in cipher2)
    bit_diffs = sum(b1 != b2 for b1, b2 in zip(bits1, bits2))
    return (bit_diffs / len(bits1)) * 100


# === MAIN TEST FILE ===
file_path = "txtfile.txt"  # or .docx / .pdf
data = read_file(file_path)

# Modify one byte
data_modified = bytearray(data)
data_modified[0] ^= 0b00000001  # Flip the lowest bit

key = b"1234567890abcdef"  # 16 bytes AES key used in encryption

cipher1 = encrypt_aes(data, key)
cipher2 = encrypt_aes(data_modified, key)

print(f"Avalanche Effect: {avalanche_effect(cipher1, cipher2):.2f}%")
print(f"Entropy (original): {calculate_entropy(data):.4f}")
print(f"Entropy (cipher): {calculate_entropy(cipher1):.4f}")
print(f"Byte Diff %: {byte_diff_percentage(cipher1, cipher2):.2f}%")
print(f"Compression Ratio (cipher): {compression_ratio(cipher1):.4f}")
