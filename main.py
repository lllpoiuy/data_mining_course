import argparse
def main():
    parser = argparse.ArgumentParser(description="Data Mining Course Script")
    parser.add_argument('--type', choices=['train', 'test'], required=True, help='Type of operation: train or test')
    parser.add_argument('--path', required=True, help='Path to the dataset')
    parser.add_argument('--num', type=int, required=False, help='An integer parameter')
    args = parser.parse_args()

    print(f"Operation type: {args.type}")
    print(f"Dataset path: {args.path}")

if __name__ == "__main__":
    main()