import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import time
from datetime import datetime

def get_date_time():
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return dt_string

print("Program start: ", get_date_time())

torch.manual_seed(73)

train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

batch_size = 64

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
        # the model uses the square activation function
        x = x * x
        # flattening while keeping the batch axis
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        return x


def train(model, train_loader, criterion, optimizer, n_epochs=10):
    # model in training mode
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

        # calculate average losses
        train_loss = train_loss / len(train_loader)

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    
    # model in evaluation mode
    model.eval()
    return model

startTimeNTSModel = time.time()

model = ConvNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model = train(model, train_loader, criterion, optimizer, 10)

print(f'Processing time for model with no tenseal: {time.time() - startTimeNTSModel}')

startTimeNTS = time.time()

def test(model, test_loader, criterion):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    # model in evaluation mode
    model.eval()

    for data, target in test_loader:
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss/len(test_loader)
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

print("End time for no tenseal: ", get_date_time())

"""
It's a PyTorch-like model using operations implemented in TenSEAL.
    - .mm() method is doing the vector-matrix multiplication explained above.
    - you can use + operator to add a plain vector as a bias.
    - .conv2d_im2col() method is doing a single convolution operation.
    - .square_() just square the encrypted vector inplace.
"""

import tenseal as ts

startTimeTSModel = time.time()

class EncConvNet:
    def __init__(self, torch_nn):
        # Initialize weights and biases for convolution and fully connected layers
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
        # Handle each channel in batch
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            # Apply convolution across the entire batch
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias  # Use broadcasting for bias
            enc_channels.append(y)

        # Pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        
        # Square activation for the packed vector
        enc_x.square_()

        # Fully connected layers processing for the entire batch
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias  # Matrix multiplication with bias
        enc_x.square_()
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        
        return enc_x
    
def enc_test(context, model, test_loader, criterion, kernel_shape, stride):
    # Initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for data, target in test_loader:
        # Lists to accumulate decrypted outputs for the batch
        batch_outputs = []
        
        for i in range(data.size(0)):  # Process each sample in the batch individually
            # Encode and encrypt each sample separately
            x_enc, windows_nb = ts.im2col_encoding(
                context, data[i].view(28, 28).tolist(), kernel_shape[0],
                kernel_shape[1], stride
            )
            
            # Encrypted evaluation using the forward method
            enc_output = model.forward(x_enc, windows_nb)
            
            # Decrypt the result
            output = enc_output.decrypt()
            batch_outputs.append(output)
        
        # Convert accumulated batch outputs to a tensor for evaluation
        output = torch.tensor(batch_outputs)
        
        # Compute loss across the batch
        loss = criterion(output, target)
        test_loss += loss.item()
        
        # Convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        
        # Compare predictions to true labels
        correct = pred.eq(target.view_as(pred))
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # Calculate and print avg test loss
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.6f}\n')

    for label in range(10):
        if class_total[label] > 0:  # Avoid division by zero
            print(
                f'Test Accuracy of {label}: {int(100 * class_correct[label] / class_total[label])}% '
                f'({int(class_correct[label])}/{int(class_total[label])})'
            )

    print(
        f'\nTest Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}% ' 
        f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})'
    )

# Load one element at a time
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
# required for encoding
kernel_shape = model.conv1.kernel_size
stride = model.conv1.stride[0]

## Encryption Parameters

# controls precision of the fractional part
bits_scale = 26

# Create TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
)

# set the scale
context.global_scale = pow(2, bits_scale)

# galois keys are required to do ciphertext rotations
context.generate_galois_keys()

enc_model = EncConvNet(model)

endTimeTSModel = time.time()

print(f'Processing time for tenseal model: {endTimeTSModel - startTimeTSModel}')

startTimeTS = time.time()

enc_test(context, enc_model, test_loader, criterion, kernel_shape, stride)

endTimeTS = time.time()

print(f'Processing time for tenseal testing: {endTimeTS - startTimeTS}')

print("End time for with tenseal: ", get_date_time())