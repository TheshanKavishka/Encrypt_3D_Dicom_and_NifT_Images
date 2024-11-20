import nibabel as nib
import numpy as np
import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import hashlib

def generate_key(text, salt):
    # Concatenate the text and salt
    text_with_salt = text + salt
    # Use SHA-256 to hash the text with salt
    key = hashlib.sha256(text_with_salt.encode()).digest()
    return key

# Function to convert a hexadecimal key to a 256-bit binary representation
def hex_to_256bit_binary(hex_key):
    return ''.join(format(int(c, 16), '04b') for c in hex_key)

# Function to convert 16-bit binary string to integer
def binary_to_int(b):
    return int(b, 2)

# Function to perform XOR on two binary strings of equal length
def xor_binary(bin1, bin2):
    return ''.join('1' if b1 != b2 else '0' for b1, b2 in zip(bin1, bin2))

def reverse_substitute_image_scrambling_3d(substituted_image, seed):
    np.random.seed(seed)
    flat_substituted = substituted_image.flatten()
    indices = np.arange(len(flat_substituted))
    np.random.shuffle(indices)
    inverse_indices = np.argsort(indices)
    original_image = flat_substituted[inverse_indices].reshape(substituted_image.shape)
    return original_image

def reverse_substitute_image_permutation_3d(substituted_image, permutation):
    flat_substituted = substituted_image.flatten()
    inverse_permutation = np.argsort(permutation)
    original_image = flat_substituted[inverse_permutation].reshape(substituted_image.shape)
    return original_image

# Function to convert hexadecimal key to bytes
def hex_to_bytes(hex_key):
    return bytes.fromhex(hex_key)

# Function to initialize AES cipher
def init_aes_cipher(key, nonce):
    # Ensure key length is 32 bytes (256 bits) using SHA-256
    key_bytes = hashlib.sha256(key).digest()
    return AES.new(key_bytes, AES.MODE_EAX, nonce=nonce)

def process_nii_decryption(input_file_path, output_file_path, permutation, seed, key):
    encrypted_nii = nib.load(os.path.join(input_file_path, 'encrypted_image.nii'))
    encrypted_data = encrypted_nii.get_fdata()
    
    # Convert the hexadecimal key to bytes
    key_bytes = hex_to_bytes(key.hex())
    key_binary = hex_to_256bit_binary(key.hex())
    
    # Initialize an empty 3D array for the decrypted data
    decrypted_3d = np.zeros_like(encrypted_data)
    
    for z in range(encrypted_data.shape[2]):
        # Get the 2D slice
        encrypted_2d = encrypted_data[:, :, z]
        
        # Read the combined data from the file
        slice_filename = os.path.join(input_file_path, f'slice_{z+1}.enc')
        with open(slice_filename, 'rb') as f:
            combined = f.read()

        # Extract nonce, tag, and encrypted bytes
        nonce = combined[:16]
        tag = combined[16:32]
        encrypted_bytes = combined[32:]

        # Initialize AES cipher
        cipher = init_aes_cipher(key_bytes, nonce)

        # Decrypt the slice data
        padded_slice_bytes = cipher.decrypt_and_verify(encrypted_bytes, tag)

        # Unpad the decrypted bytes
        slice_bytes = unpad(padded_slice_bytes, AES.block_size)

        # Convert the bytes back to the original 2D shape
        decrypted_2d = np.frombuffer(slice_bytes, dtype=np.uint16).reshape(encrypted_2d.shape)

        # Convert the decrypted integers back to binary string
        decrypted_binary = ''.join(format(pixel, '016b') for pixel in decrypted_2d.flatten())

        # Perform XOR decryption on the slice data and the key
        decrypted_binary = xor_binary(decrypted_binary, key_binary * (len(decrypted_binary) // len(key_binary) + 1))

        # Convert the decrypted binary string back to integers
        decrypted_ints = [binary_to_int(decrypted_binary[j:j+16]) for j in range(0, len(decrypted_binary), 16)]

        # Reshape the decrypted integers to their original 2D shape
        decrypted_2d = np.array(decrypted_ints).reshape(encrypted_2d.shape)

        # Store the decrypted 2D slice in the 3D array
        decrypted_3d[:, :, z] = decrypted_2d

    # Create a new 3D NIfTI image with the decrypted data
    decrypted_nii = nib.Nifti1Image(decrypted_3d, affine=encrypted_nii.affine)
            
    substituted_image_data = decrypted_nii.get_fdata()

    original_image_data = reverse_substitute_image_scrambling_3d(substituted_image_data, seed)
    original_image_data = reverse_substitute_image_permutation_3d(original_image_data, permutation)
        
    original_nii_image = nib.Nifti1Image(original_image_data, decrypted_nii.affine)

    # Save the new 3D NIfTI image
    output_file_path = os.path.join(output_file_path, 'decrypted_3d_image.nii')
    nib.save(original_nii_image, output_file_path)
    print(f"Decrypted 3D NIfTI image saved to: {output_file_path}")


input_folder = 'Final Encryption/'
output_folder = 'Final Decryption/'
seed = 42  # Example seed for block scrambling
text = input("Enter text to derive key: ")
salt = input("Enter salt (in hex): ")
key = generate_key(text, salt)
print("Derived Key (in hex):", key.hex())

# Process the DICOM series and save encrypted images (block scrambling and DNA addition)
permutation_file_path = 'permutation.txt'

# Read the permutation array from the text file
permutation = np.loadtxt(permutation_file_path, dtype=int)

process_nii_decryption(input_folder, output_folder, permutation, seed, key)
