import os
import numpy as np
import pydicom
from PIL import Image
from Crypto.Cipher import AES
import hashlib

# AES decryption function
def aes_decrypt(encrypted_data, key, nonce, tag):
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    decrypted_data = cipher.decrypt_and_verify(encrypted_data, tag)
    return decrypted_data

def generate_key(text, salt):
    # Concatenate the text and salt
    text_with_salt = text + salt
    # Use SHA-256 to hash the text with salt
    key = hashlib.sha256(text_with_salt.encode()).digest()
    return key


def reverse_substitute_image_scrambling(substituted_image, seed):
    np.random.seed(seed)
    flat_substituted = substituted_image.flatten()
    indices = np.arange(len(flat_substituted))
    np.random.shuffle(indices)
    inverse_indices = np.argsort(indices)
    original_image = flat_substituted[inverse_indices].reshape(substituted_image.shape)
    return original_image

def reverse_substitute_image_permutation(substituted_image, permutation):
    flat_substituted = substituted_image.flatten()
    inverse_permutation = np.argsort(permutation)
    original_image = flat_substituted[inverse_permutation].reshape(substituted_image.shape)
    return original_image

# Function to convert a hexadecimal key to a 256-bit binary representation
def hex_to_256bit_binary(hex_key):
    return ''.join(format(int(c, 16), '04b') for c in hex_key)

# Function to convert 16-bit binary string to integer
def binary_to_int(b):
    return int(b, 2)

# Function to perform XOR on two binary strings of equal length
def xor_binary(bin1, bin2):
    return ''.join('1' if b1 != b2 else '0' for b1, b2 in zip(bin1, bin2))

def process_dicom_series_block_scrambling(input_folder, output_folder, permutation, block_size, seed, key):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".dcm"):
            filepath = os.path.join(input_folder, filename)
            dicom_image = pydicom.dcmread(filepath)

            if dicom_image.file_meta.TransferSyntaxUID.is_compressed:
                dicom_image.decompress()

            encrypted_data = dicom_image.PixelData

            # Load nonce, tag, and shape for the current file
            nonce_tag_shape_file = os.path.join(input_folder, filename + '_nonce_tag_shape.npy')
            data = np.load(nonce_tag_shape_file, allow_pickle=True).item()
            nonce, tag, shape = data['nonce'], data['tag'], data['shape']

            decrypted_data = aes_decrypt(encrypted_data, key, nonce, tag)
            dicom_image.PixelData = np.frombuffer(decrypted_data, dtype=np.uint16).reshape(shape).tobytes()

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
                    
                    # dicom_image.PixelData = xored_image.tobytes()
            image_data = xored_image_array    
            # substituted_image_data = image_data
            original_image_data = reverse_substitute_image_scrambling(image_data, seed)
            original_image_data = reverse_substitute_image_permutation(original_image_data, permutation)

            dicom_image.PixelData = original_image_data.tobytes()
            output_filepath = os.path.join(output_folder, filename)
            dicom_image.save_as(output_filepath)


input_folder = 'final_encryption/'
output_folder = 'final_decryption/'
block_size = 8  # Example block size
seed = 42  # Example seed for block scrambling
text = input("Enter text to derive key: ")
salt = input("Enter salt (in hex): ")
key = generate_key(text, salt)
print("Derived Key (in hex):", key.hex())

# Process the DICOM series and save encrypted images (block scrambling and DNA addition)
permutation_file_path = 'permutation.txt'

# Read the permutation array from the text file
permutation = np.loadtxt(permutation_file_path, dtype=int)

process_dicom_series_block_scrambling(input_folder, output_folder, permutation, block_size, seed, key)







