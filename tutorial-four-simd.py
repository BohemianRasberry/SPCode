import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import time
import tenseal as ts

torch.manual_seed(73)

# Load the MNIST dataset
train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

batch_size = 64

# Data loaders for training and testing
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

class ConvNet(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        super(ConvNet, self).__init__()        
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = torch.nn.Linear(256, hidden)
        self.fc2 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = self.conv1(x)
        # The model uses the square activation function
        x = x * x
        # Flattening while keeping the batch axis
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        return x

def train(model, train_loader, criterion, optimizer, n_epochs=10):
    # Model in training mode
    model.train()
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Calculate average losses
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    
    # Model in evaluation mode
    model.eval()
    return model

# Timing the model training
startTimeNTSModel = time.time()

model = ConvNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model = train(model, train_loader, criterion, optimizer, 10)

print(f'Processing time for model with no TenSEAL: {time.time() - startTimeNTSModel}')

# Timing the testing process
startTimeNTS = time.time()

def test(model, test_loader, criterion):
    # Initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    # Model in evaluation mode
    model.eval()

    for data, target in test_loader:
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()
        # Convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # Compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # Calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # Calculate and print avg test loss
    test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {test_loss:.6f}\n')

    for label in range(10):
        print(
            f'Test Accuracy of {label}: {int(100 * class_correct[label] / class_total[label])}% '
            f'({int(np.sum(class_correct[label]))}/{int(np.sum(class_total[label]))})'
        )

    print(
        f'\nTest Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}% ' 
        f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})'
    )
    
test(model, test_loader, criterion)

endTimeNTS = time.time()

print(f'The total time for no TenSEAL testing: {endTimeNTS - startTimeNTS}')

"""
This part uses operations implemented in TenSEAL for homomorphic encryption.
- The .mm() method does vector-matrix multiplication.
- You can use the + operator to add a plain vector as a bias.
- The .conv2d_im2col() method performs a single convolution operation.
- The .square_() method squares the encrypted vector in place.
"""

startTimeTSModel = time.time()

class EncConvNet:
    def __init__(self, torch_nn):
        # Extract weights and biases from the trained PyTorch model
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()
        
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()
        
    def forward(self, enc_x, windows_nb):
        # Convolutional layer
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)  # Pack all channels

        # Square activation
        enc_x.square_()

        # Fully connected layer 1
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        enc_x.square_()

        # Fully connected layer 2
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def is_power_of_two(n: int) -> bool:
    """Check if n is a power of two."""
    return n & (n - 1) == 0

def enc_rotate_and_sum(ciphertext: ts.CKKSVector) -> ts.CKKSVector:
    """Return a ciphertext where each entry contains the sum of all entries in the input ciphertext."""
    n = ciphertext.size()  # Get the number of elements in CKKSVector
    assert is_power_of_two(n), "The length of the ciphertext must be a power of two."
    
    # Create a sum vector initialized to zero using the context from the ciphertext
    sum_vector = ts.CKKSVector(ciphertext.context, [0] * n)

    # Pairwise summation using rotations
    shift = n // 2
    while shift > 0:
        rotated = ciphertext.rotate(shift)  # Rotate ciphertext
        sum_vector += rotated  # Add rotated ciphertext to sum_vector
        shift //= 2
    
    sum_vector += ciphertext  # Add the original vector to the result

    return sum_vector

def enc_test(context, model, test_loader, criterion, kernel_shape, stride):
    # Initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for data, target in test_loader:
        # Encoding and encryption
        x_enc, windows_nb = ts.im2col_encoding(
            context, data.view(28, 28).tolist(), kernel_shape[0],
            kernel_shape[1], stride
        )

        # Encrypted evaluation
        enc_output = model(x_enc, windows_nb)

        # Check the size of enc_output and pad if necessary
        n = enc_output.size()
        if not is_power_of_two(n):
            # Calculate the next power of two
            next_power_of_two = 1 << (n - 1).bit_length()  # Calculate the next power of two
            
            # Calculate the required padding size
            padding_size = next_power_of_two - n
            
            # Create a new CKKSVector for padding filled with zeros
            padded_data = [0] * padding_size
            
            # Create a new CKKSVector for the padded data
            enc_output_padded = ts.CKKSVector(context, padded_data)
            
            # To combine, we'll have to make sure the sizes are compatible
            # Instead of concatenating, create a new padded CKKSVector of the correct size
            # by using enc_output's data and then appending the padded data.
            enc_output = ts.CKKSVector(context, enc_output.decrypt() + padded_data)  # Decrypt, add padding, and re-encrypt

        # Encrypted rotate and sum
        enc_output = enc_rotate_and_sum(enc_output)

        # Decryption of result
        output = enc_output.decrypt()
        output = torch.tensor(output).view(1, -1)

        # Compute loss
        loss = criterion(output, target)
        test_loss += loss.item()

        # Convert output probabilities to predicted class
        _, pred = torch.max(output, 1)

        # Compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))

        # Calculate test accuracy for each object class
        label = target.data[0]
        class_correct[label] += correct.item()
        class_total[label] += 1

    # Calculate and print avg test loss
    test_loss = test_loss / sum(class_total)
    print(f'Test Loss: {test_loss:.6f}\n')

    for label in range(10):
        print(
            f'Test Accuracy of {label}: {int(100 * class_correct[label] / class_total[label])}% '
            f'({int(np.sum(class_correct[label]))}/{int(np.sum(class_total[label]))})'
        )

    print(
        f'\nTest Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}% ' 
        f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})'
    )

# Load one element at a time for encrypted testing
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

# Required for encoding
kernel_shape = model.conv1.kernel_size
stride = model.conv1.stride[0]

## Encryption Parameters

# Controls precision of the fractional part
bits_scale = 26

# Create TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
)

# Set the scale
context.global_scale = pow(2, bits_scale)

# Galois keys are required to do ciphertext rotations
context.generate_galois_keys()

# Initialize the encrypted model
enc_model = EncConvNet(model)

endTimeTSModel = time.time()

print(f'Processing time for TenSEAL model: {endTimeTSModel - startTimeTSModel}')

# Measure processing time for TenSEAL model
startTimeTS = time.time()
enc_test(context, enc_model, test_loader, criterion, kernel_shape, stride)
endTimeTS = time.time()

print(f'Processing time for TenSEAL model: {endTimeTS - startTimeTS}')
