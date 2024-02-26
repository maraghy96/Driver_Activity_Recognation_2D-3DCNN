import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)  # Initialy kernel was 3 and 64 instead of 32
        self.bn1 = nn.BatchNorm3d(32)
        self.pool = nn.MaxPool3d(2)
        #self.conv2 = nn.Conv3d(32, 128, kernel_size=1, padding=1)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)

        # Update the following line based on the actual output size
        self.fc1 = nn.Linear(64 * 4 * 56 * 56, 256)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, num_classes)
        # Calculate the correct number of flattened features from the output of the last conv layer
        # Assuming the output of the last pooling layer is [batch_size, 128, depth, height, width]
        # You will need to update the following line based on the actual size of the output from the last pooling layer
       # self.fc1 = nn.Linear(128 * 4 * 56 * 56, 512)
        #self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the tensor for the linear layer
        x = x.view(x.size(0), -1)  # Flatten all dimensions except the batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
