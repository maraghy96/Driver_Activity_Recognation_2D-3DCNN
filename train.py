import time
from tqdm import tqdm
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device=None):
    since = time.time()

    val_acc_history = []
    best_acc = 0.0
    confusion_matrix_total = None
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            accumulation_steps = 4  
            optimizer.zero_grad()


            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=f'Epoch {epoch+1} {phase} Progress'):


                # If the number of dimensions is 6, the permute call will fail.
                # Check if the number of dimensions is 5 before permuting.
                if inputs.dim() == 5:
                    inputs = inputs.permute(0, 2, 1, 3, 4)  # Adjusting the dimensions

                else:
                    print("Unexpected number of dimensions in the input tensor.")

                inputs = inputs.to(device)
                labels = labels.to(device)
                #print(f"Input shape: {inputs.shape}")
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Confirming the range and type of labels


                # Confirming the range of labels is within [0, num_classes-1]
                num_classes = 20  # Change this to your actual number of classes
                if labels.min() < 0 or labels.max() >= num_classes:
                    print(f"Error: Labels out of range. Should be 0 to {num_classes-1}")
                    break  # Remove this line if you want to continue despite the error

                # Assuming outputs is the model's predictions
                outputs = model(inputs)
                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                      outputs = model(inputs)
                      loss = criterion(outputs, labels)

                      _, preds = torch.max(outputs, 1)

                      # Backward + optimize only if in training phase
                      if phase == 'train':
                          loss.backward()
                          optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = (running_corrects.double() / len(dataloaders[phase].dataset) ) *100

            print('{} Loss: {:.4f} Acc: {:.4f} % '.format(phase, epoch_loss, epoch_acc))

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                val_acc_history.append(epoch_acc)
                if confusion_matrix_total is None:
                    confusion_matrix_total = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
                else:
                    confusion_matrix_total += confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model, val_acc_history, confusion_matrix_total
