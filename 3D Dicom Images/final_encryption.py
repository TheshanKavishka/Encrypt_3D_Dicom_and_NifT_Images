import os
import numpy as np
import pydicom
from PIL import Image
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import hashlib

def generate_key(text, salt):
    # Concatenate the text and salt
    text_with_salt = text + salt
    # Use SHA-256 to hash the text with salt
    key = hashlib.sha256(text_with_salt.encode()).digest()
    return key

def generate_salt(length=16):
    # Generate a random salt
    salt = get_random_bytes(length)
    return salt

# AES encryption function
def aes_encrypt(data, key):
    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(data.tobytes())
    return ciphertext, cipher.nonce, tag

# Function to substitute image using AES encryption
def substitute_image_aes(image, key):
    # Flatten image data to encrypt the entire pixel array
    flattened_data = image.flatten()
    encrypted_data, nonce, tag = aes_encrypt(flattened_data, key)
    return encrypted_data, nonce, tag, image.shape

# Function to convert a hexadecimal key to a 256-bit binary representation
def hex_to_256bit_binary(hex_key):
    return ''.join(format(int(c, 16), '04b') for c in hex_key)

# Function to convert 16-bit binary string to integer
def binary_to_int(b):
    return int(b, 2)

# Function to perform XOR on two binary strings of equal length
def xor_binary(bin1, bin2):
    return ''.join('1' if b1 != b2 else '0' for b1, b2 in zip(bin1, bin2))

def substitute_image_permutation(image, permutation):
    flat_image = image.flatten()
    substituted_image = flat_image[permutation].reshape(image.shape)
    return substituted_image

def generate_permutation(size):
    permutation = np.random.permutation(size)
    return permutation

def substitute_image_scrambling(image, seed):
    np.random.seed(seed)
    flat_image = image.flatten()
    np.random.shuffle(flat_image)
    substituted_image = flat_image.reshape(image.shape)
    return substituted_image


def process_dicom_series_combine(input_folder, output_folder, permutation, block_size, seed, key):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".dcm"):
            filepath = os.path.join(input_folder, filename)
            dicom_image = pydicom.dcmread(filepath)

            if dicom_image.file_meta.TransferSyntaxUID.is_compressed:
                dicom_image.decompress()

            image_data = dicom_image.pixel_array
            substituted_image_data = substitute_image_permutation(image_data, permutation)
            substituted_image_data = substitute_image_scrambling(substituted_image_data, seed)
            
            dicom_image.PixelData = substituted_image_data.tobytes()
        
            key_binary = hex_to_256bit_binary(key.hex())
            key_length = len(key_binary)
            image_array = dicom_image.pixel_array
            rows, cols = image_array.shape
            block_size = 8
            xored_image_array = np.zeros_like(image_array)


            for i in range(0, rows, block_size):
                for j in range(0, cols, block_size):
                    segment = image_array[i:i+block_size, j:j+block_size]
                    segment_binary = ''.join(format(pixel, '016b') for row in segment for pixel in row)

                    # Repeat the key to match the length of the segment binary string if necessary
                    extended_key_binary = (key_binary * ((len(segment_binary) // key_length) + 1))[:len(segment_binary)]

                    xored_segment_binary = xor_binary(segment_binary, extended_key_binary)
                    xored_segment = np.array([binary_to_int(xored_segment_binary[k:k+16]) for k in range(0, len(xored_segment_binary), 16)])
                    xored_segment = xored_segment.reshape((block_size, block_size))

                    xored_image_array[i:i+block_size, j:j+block_size] = xored_segment

                    
            # xored_image = Image.fromarray(xored_image_array)

            image_data = xored_image_array
            encrypted_data, nonce, tag, shape = substitute_image_aes(image_data, key)

            output_filepath = os.path.join(output_folder, filename)
            dicom_image.PixelData = encrypted_data  # Store encrypted data as PixelData
            dicom_image.save_as(output_filepath)

            # Save nonce, tag, and shape for each file separately
            nonce_tag_shape_file = os.path.join(output_folder, filename + '_nonce_tag_shape.npy')
            np.save(nonce_tag_shape_file, {'nonce': nonce, 'tag': tag, 'shape': shape})    

image_size = 512 * 512  # Example for 512x512 images
permutation = generate_permutation(image_size)
seed = 42  # Example seed for scrambling
block_size = 8  # Example block size
input_folder = 'series-000001/'
output_folder = 'final_encryption/'
text = input("Enter text to derive key: ")
salt = generate_salt()
key = generate_key(text, salt.hex())
print("Generated Key (in hex):", key.hex())
print("Generated Salt (in hex):", salt.hex())
permutation_file_path = 'permutation.txt'
np.savetxt(permutation_file_path, permutation, fmt='%d')

process_dicom_series_combine(input_folder, output_folder, permutation, block_size, seed, key)

