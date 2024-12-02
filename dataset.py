"""
This file provides code for loading the Platypus dataset and doing sampling on the data. This data can then be used by other files (ie. for finetuning).
"""
from datasets import load_dataset
import random

class PlatypusDataset:
    def __init__(self):
        self.data = load_dataset("garage-bAInd/Open-Platypus", split='train')
        train_val_split = self.data.train_test_split(test_size=0.2)
        self.train_dataset = train_val_split['train']
        self.val_dataset = train_val_split['test']

    def get_length(self):
        return len(self.data)

    def get_item(self, index):
        return self.data[index]

    def random_sample(self):
        # Random sampling using filter
        sampled_dataset = self.data.filter(lambda x: random.random() < 0.1)  # ~10% sample
        return sampled_dataset
    
    def get_train_data_without_source(self, source_name):
        return [item for item in self.train_dataset if item['data_source'] != source_name]
    
    def get_val_data_without_source(self, source_name):
        return [item for item in self.val_dataset if item['data_source'] != source_name]

    
    def get_data_without_source(self, source_name):
        '''
        Return the subset of the dataset that doesn't include points from source_name
        '''
        return [item for item in self.data if item['data_source'] != source_name]
    
    def get_data_from_source(self, source_name):
        '''
        Return the subset of the dataset from given source_name
        '''
        return [item for item in self.data if item['data_source'] == source_name]
    
    def get_sources(self):
        sources = set()
        
        for item in self.data:
            item_source = item['data_source']
            if item_source in sources:
                continue
            sources.add(item_source)
            
        return list(sources)
        

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
    
    data_sources = dataset.get_sources()
    print(data_sources)
    
    for source in data_sources:
        train = dataset.get_train_data_without_source(source)
        test = dataset.get_val_data_without_source(source)
        
        # print(source)
        # print("Train: " + str(len(train)))
        # print("Test: " + str(len(test)))

if __name__ == "__main__":
    main()


