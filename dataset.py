"""
This file provides code for loading the Platypus dataset and doing sampling on the data. This data can then be used by other files (ie. for finetuning).
"""
from datasets import load_dataset
import random

class PlatypusDataset:
    def __init__(self):
        self.data = load_dataset("garage-bAInd/Open-Platypus", split='train')

    def get_length(self):
        return len(self.data)

    def get_item(self, index):
        return self.data[index]

    def random_sample(self):
        # Random sampling using filter
        sampled_dataset = self.data.filter(lambda x: random.random() < 0.1)  # ~10% sample
        return sampled_dataset

def main():
    """
    Main function to demonstrate Dataset class usage.
    """
    # Create dataset instance
    dataset = PlatypusDataset()

    # Get and print the length
    dataset_size = dataset.get_length()
    print(f"Dataset size: {dataset_size}")

    print(dataset.get_item(0))

if __name__ == "__main__":
    main()


