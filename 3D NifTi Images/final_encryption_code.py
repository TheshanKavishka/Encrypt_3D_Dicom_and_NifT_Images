import nibabel as nib
import numpy as np
import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes
import hashlib

# Function to convert hexadecimal key to bytes
def hex_to_bytes(hex_key):
    return bytes.fromhex(hex_key)

# Function to initialize AES cipher
def init_aes_cipher(key):
    # Ensure key length is 32 bytes (256 bits) using SHA-256
    key_bytes = hashlib.sha256(key).digest()
    return AES.new(key_bytes, AES.MODE_EAX)

# Function to convert a hexadecimal key to a 256-bit binary representation
def hex_to_256bit_binary(hex_key):
    return ''.join(format(int(c, 16), '04b') for c in hex_key)

# Function to convert 16-bit binary string to integer
def binary_to_int(b):
    return int(b, 2)

# Function to perform XOR on two binary strings of equal length
def xor_binary(bin1, bin2):
    return ''.join('1' if b1 != b2 else '0' for b1, b2 in zip(bin1, bin2))

def substitute_image_permutation_3d(image, permutation):
    flat_image = image.flatten()
    substituted_image = flat_image[permutation].reshape(image.shape)
    return substituted_image

def generate_permutation_3d(size):
    permutation = np.random.permutation(size)
    return permutation

def substitute_image_scrambling_3d(image, seed):
    np.random.seed(seed)
    flat_image = image.flatten()
    np.random.shuffle(flat_image)
    substituted_image = flat_image.reshape(image.shape)
    return substituted_image

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

def process_nii_image(input_filepath, output_filepath, permutation, seed, key):
    nii_image = nib.load(input_filepath)
    image_data = nii_image.get_fdata()

    substituted_image_data = substitute_image_permutation_3d(image_data, permutation)
    substituted_image_data = substitute_image_scrambling_3d(substituted_image_data, seed)
    substituted_nii_image = nib.Nifti1Image(substituted_image_data, nii_image.affine)

    os.makedirs(output_filepath, exist_ok=True)

    # Get the dimensions of the original 3D NIfTI image
    original_shape = substituted_nii_image.shape

    # Initialize an empty 3D array to store the encrypted slices
    encrypted_3d = np.zeros(original_shape)

    # Convert the 16-bit binary key to a 256-bit binary representation
    key_binary = hex_to_256bit_binary(key.hex())

    sub_img_data = substituted_nii_image.get_fdata()

    key_bytes = hex_to_bytes(key.hex())

    for z in range(sub_img_data.shape[2]):
        # Get the 2D slice
        slice_2d = sub_img_data[:, :, z]
        print(f"Slice {z} shape: {slice_2d.shape}")

        # Scale slice data to integer values (optional: adjust scaling factor as needed)
        slice_int = np.rint(slice_2d * 65535.0).astype(int)

        # Convert the slice data to a binary string
        slice_binary = ''.join(format(pixel, '016b') for pixel in slice_int.flatten())

        # Perform XOR encryption on the slice data and the key
        encrypted_binary = xor_binary(slice_binary, key_binary * (len(slice_binary) // len(key_binary) + 1))

        # Convert the encrypted binary string back to integers
        encrypted_ints = [binary_to_int(encrypted_binary[j:j+16]) for j in range(0, len(encrypted_binary), 16)]

        # Reshape the encrypted integers to their original 2D shape
        encrypted_2d = np.array(encrypted_ints).reshape(slice_int.shape)

        # Store the encrypted 2D slice in the 3D array
        encrypted_3d[:, :, z] = encrypted_2d

        # Convert the slice data to bytes
        slice_bytes = encrypted_2d.astype(np.uint16).tobytes()

        # Pad the slice bytes to make the length a multiple of AES block size (16 bytes)
        padded_slice_bytes = pad(slice_bytes, AES.block_size)

        # Initialize AES cipher (new nonce for each slice)
        cipher = init_aes_cipher(key_bytes)
        nonce = cipher.nonce

        # Perform AES encryption on the slice data
        encrypted_bytes, tag = cipher.encrypt_and_digest(padded_slice_bytes)

        # Combine nonce, tag, and encrypted bytes
        combined = nonce + tag + encrypted_bytes

        # Save the combined data to a file
        slice_filename = os.path.join(output_filepath, f'slice_{z+1}.enc')
        with open(slice_filename, 'wb') as f:
            f.write(combined)

    # Create a new 3D NIfTI image with the encrypted data
    encrypted_nii = nib.Nifti1Image(encrypted_3d, affine=nii_image.affine)
    encrypted_nii.to_filename(os.path.join(output_filepath, 'encrypted_image.nii'))


input_filepath = 'first.nii'
output_filepath = 'Final Encryption/'

img = nib.load(input_filepath)
    
dim_length = img.shape 
image_size = dim_length[0] * dim_length[1] * dim_length[2]

permutation = generate_permutation_3d(image_size)
seed = 42  # Example seed for scrambling

text = input("Enter text to derive key: ")
salt = generate_salt()
key = generate_key(text, salt.hex())
print("Generated Key (in hex):", key.hex())
print("Generated Salt (in hex):", salt.hex())
permutation_file_path = 'permutation.txt'
np.savetxt(permutation_file_path, permutation, fmt='%d')

process_nii_image(input_filepath, output_filepath, permutation, seed, key)
