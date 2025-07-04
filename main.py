import argparse
from train import train
def main():
    parser = argparse.ArgumentParser(description="Data Mining Course Script")
    parser.add_argument('--type', choices=['train', 'test'], required=True, help='Type of operation: train or test')
    parser.add_argument('--path', required=True, help='Path to the dataset')
    parser.add_argument('--num', type=int, required=False, help='An integer parameter')
    args = parser.parse_args()

    print(f"Operation type: {args.type}")
    print(f"Dataset path: {args.path}")

    if args.type == 'train':
        print("Training mode selected.")
        if args.num is not None:
            print(f"Using integer parameter: {args.num}")
            train(args.path, args.num)
        else:
            print("Training for num in [1, 12).")
            for num in range(1, 11):
                print(f"Training with num = {num}")
                train(args.path, num)

    elif args.type == 'test':
        print("Testing mode selected.")
        assert args.num is not None, "Integer parameter is required for testing."
        print(f"Testing with integer parameter: {args.num}")

if __name__ == "__main__":
    main()