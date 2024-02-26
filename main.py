import argparse
from dataset import VideoDataset
from model import Simple2DCNN  # or Simple3DCNN
from train import train_model
from torch.utils.data import DataLoader

def main(args):
    # Initialize dataset
    dataset = VideoDataset(directory=args.data_dir, transform=None, model_type=args.model_type)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    if args.model_type == '2DCNN':
        model = Simple2DCNN(num_classes=20)
    elif args.model_type == '3DCNN':
        model = Simple3DCNN(num_classes=20)
    else:
        raise ValueError("Unsupported model type")
    
    # Train model
    train_model(model, dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--model_type', type=str, choices=['2DCNN', '3DCNN'], required=True, help='Type of the model to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()
    
    main(args)
