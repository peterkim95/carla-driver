import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tutorialnet import TutorialNet
from pilotnet import PilotNet
from dataset import DrivingDataset
from util import get_args, save_checkpoint, makedirs


def main():
    args = get_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    # TODO: when input size doesn't vary, cudnn finds best algorithm?
    torch.backends.cudnn.benchmark = True 

    print(f'device: {device}')

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((200, 66)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Set data generators
    train_set = DrivingDataset(args.train, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=6)

    val_set = DrivingDataset(args.val, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=6)
    print(f'# train examples: {len(train_set)}, # val examples: {len(val_set)}')

    # Init neural net
    net = PilotNet()
    net.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # Init Tensorboard writer
    writer = SummaryWriter()

    is_best = True
    best_val_loss = float('inf')
    makedirs(args.checkpoints_path)

    # TODO: Look at keras progress bar code to get more inspiration
    progress_bar_format = '{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}{postfix}]{bar:-10b}' 

    # Perform dark magic
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')

        # Train
        train_loss = 0.0
        train_loop = tqdm(enumerate(train_loader), total=len(train_loader), unit='step', bar_format=progress_bar_format)
        for i, (local_batch, local_labels) in train_loop:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            optimizer.zero_grad()

            outputs = net(local_batch)
            loss = criterion(outputs, local_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Update progress bar
            train_loop.set_postfix(loss=train_loss / (i + 1))

        # Validate
        val_loss = 0.0
        val_loop = tqdm(enumerate(val_loader), total=len(val_loader), unit='step', bar_format=progress_bar_format)
        with torch.set_grad_enabled(False):
            for i, (local_batch, local_labels) in val_loop:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                outputs = net(local_batch)
                loss = criterion(outputs, local_labels)
                val_loss += loss.item()

                val_loop.set_postfix(val_loss=val_loss / (i + 1))

        # Save best model every epoch
        is_best = val_loss < best_val_loss
        if is_best:
            print(f'Saving new best model: val loss improved from {best_val_loss:.3f} to {val_loss:.3f}')
            save_checkpoint(net.state_dict())
            best_val_loss = min(val_loss, best_val_loss)

        # Log to Tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

    print('Finished training')

    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
