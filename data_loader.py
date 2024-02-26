#Data Loaders
#model intialization for 3D CNN
train_dataset_3d = VideoDataset(directory='/content/train_data', transform=transform, model_type='3DCNN')
test_dataset_3d = VideoDataset(directory='/content/test_data', transform=transform, model_type='3DCNN')

# For 2D CNN
#train_dataset_2d = VideoDataset(directory='/content/train_data', transform=transform, model_type='2DCNN')
#test_dataset_2d = VideoDataset(directory='/content/test_data', transform=transform, model_type='2DCNN')


# Create DataLoader instances (choose 2DCNN or 3DCNN as model_type)
train_loader = DataLoader(train_dataset_3d, batch_size=8, shuffle=True, num_workers=2)
test_loader = DataLoader(train_dataset_3d, batch_size=8, shuffle=False, num_workers=2)


dataloaders = {
    'train': train_loader,
    'val': test_loader
}
