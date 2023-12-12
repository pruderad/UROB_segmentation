import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from fastseg import MobileV3Small
from fastseg.image.colorize import colorize, blend


model = MobileV3Small.from_pretrained()
model

# Open image from file and resize it to lower memory footprint
img = Image.open('image.png').resize((1024, 512))

# Change the class from PIL.Image into numpy array
img_np = np.asarray(img)

# Create torch tensor from numpy array and add dimension representing batchsize. Also change dtype to float as it is required by torch
x = torch.tensor(img_np).unsqueeze(0).float()

# Transpose dimension of tensor so it respects the torch convention: Batch Size x Number of Classes x Height x Width
x = x.permute(0, 3, 1, 2)

# Normalize data
x = (x / 255) * 2 - 1

# Forward pass, input image x and return output probabilities for each pixel and each class along each image in batch size
output = model(x)

# Output in 
print("Following is for the first pixel [0,0] of first image in batch [0]: \n")
print('Logits: \n', output[0,:,0,0], '\n')
print('Probabilities: \n', output.softmax(dim=1)[0,:,0,0], '\n')
print('Prediction: \n', output.argmax(dim=1)[0,0,0], '\n')

# Calculation of final segmentation prediction from class probabilities along dimension 1
# detach.cpu.numpy transfer tensor from torch to computational graph-detached, to cpu memory and to numpy array instead of tensor
seg_np = output.argmax(dim=1)[0].detach().cpu().numpy()

# Function from fastseg to visualize images and output segmentation
seg_img = colorize(seg_np) # <---- input is numpy, output is PIL.Image
blended_img = blend(img, seg_img) # <---- input is PIL.Image in both arguments

# Concatenate images for simultaneous view
new_array = np.concatenate((np.asarray(blended_img), np.asarray(seg_img)), axis=1)

# Show image from PIL.Image class
combination = Image.fromarray(new_array)
# combination.show()

CE = torch.nn.CrossEntropyLoss(reduction="none", weight=None)

# Get final prediction with argmax
labels = output.argmax(dim=1)


# Initialize model, can be from pretrained version (prefered). Here it is for educational purposes
num_classes = 19
model = MobileV3Small(num_classes=num_classes)

# Set up model to training mode (some layers are designed to behave differently during learning and during inference - batch norm for example.)
# Always learn model in training mode
model.train()

# Set up optimizer to automatically update weights with respect to computed loss and negative of gradient
# Regularization weight decay - analogy with remembering the exam questions
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Multiple iterations (epochs)
for e in range(100):
    
    # Forward pass of model. Input image x and output as per-pixel probabilities, per-image in batch
    # output dimensions: Batch Size x Class probs x H x W
    output = model(x)
    
    # Calculation of Loss function, we use pytorch implementations of Cross entropy (softmax + negative log-likelihood)
    loss = CE(output, labels)
    
    # Print loss and metric Intersection-over-union to monitor model's performance during the training
    # Why there is non-zero loss when learning on the self-produced labels?
    print(f'Epoch: {e:03d}', f'Loss: {loss.mean().item():.4f}')
    
    # This step is the most important. On the backend, Torch will accumulate gradients along the performed operations and keeps it in the memory
    # After calling backward(), the gradients are recomputed for specific forward pass and the model accumulates gradients with respect to the loss
    loss.mean().backward()
    
    # After we compute the gradients from backward(), each weight in the model will have the .grad value.
    # Optimizer will then use the gradient and learning rate to update the weights
    optimizer.step()
    
    # Test if the models has accumulated gradients and therefore "learn something"
    if e == 0:
        print("Gradient in the last layer on specific weights: ", model.last.weight[0,0,0,0])
        
    # Clean already used gradients to start over in the new iteration
    optimizer.zero_grad()

    # Visualization of model's output at every iterations
    seg_np = output.argmax(dim=1)[0].detach().cpu().numpy()
    seg_img = colorize(seg_np)
    seg_img.save(f'overfitting/{e:03d}.png')


# Saving weights
torch.save(model.state_dict(), 'weights/model.pth')