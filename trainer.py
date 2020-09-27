import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import Net
from dataset import generate_labels, generate_partition, Dataset
from utils import get_args, save_checkpoint, makedirs


def main():
    args = get_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    torch.backends.cudnn.benchmark = True

    print(f'device: {device}')

    # TODO: How to handle multi-epsiode data reading?
    partition = generate_partition() # e.g. {'train': ['id-1', 'id-2', 'id-3'], 'val': ['id-4']}
    labels = generate_labels() # load from episode label dict

    # Set data generators
    train_set = Dataset(partition['train'], labels)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=6)

    val_set = Dataset(partition['val'], labels)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=6)

    # Init neural net
    net = Net()
    net.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Init Tensorboard writer
    writer = SummaryWriter()

    is_best = True
    best_val_loss = 2 ** 1000
    makedirs(args.checkpoints_path)

    # Perform dark magic
    for epoch in range(args.epochs):
        # Train
        train_loss = 0.0
        for i, (local_batch, local_labels) in enumerate(train_loader):
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            optimizer.zero_grad()

            outputs = net(local_batch)
            loss = criterion(outputs, local_labels)
            loss.backward()
            optimizer.step()

            # print stats
            train_loss += loss.item()
            # if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item() / (i + 1)))

        # Validate
        val_loss = 0.0
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in val_loader:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                outputs = net(local_batch)
                loss = criterion(outputs, local_labels)
                val_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch + 1}: train loss = {train_loss:.3f}, val loss = {val_loss:.3f}')
        
        # Save best model every epoch
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        if is_best:
            save_checkpoint(net.state_dict())

        # Log to Tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

    print('Finished training')

    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
