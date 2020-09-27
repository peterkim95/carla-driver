from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='trainer')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()
    return args