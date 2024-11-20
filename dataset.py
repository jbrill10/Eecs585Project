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
    
    # for source in data_sources:
    #     ablated_dataset = dataset.get_data_without_source(source)
    #     source_dataset = dataset.get_data_from_source(source)
        
    #     print(len(ablated_dataset) + len(source_dataset))
        
    
    

if __name__ == "__main__":
    main()


