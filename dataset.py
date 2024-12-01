"""
This file provides code for loading the Platypus dataset and doing sampling on the data. This data can then be used by other files (ie. for finetuning).
"""
from datasets import load_dataset
import random
from collections import defaultdict
import time
import cProfile
import pstats
import io
import tracemalloc
from functools import wraps


## DECORATE A FUNCTION WITH THIS TO MEASURE RUNTIME, CPU USAGE, MEMORY USAGE
def track_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start tracking memory
        tracemalloc.start()

        # Profile CPU usage
        profiler = cProfile.Profile()
        profiler.enable()

        # Measure runtime
        start = time.time()
        try:
            # Call the original function
            result = func(*args, **kwargs)
        finally:
            # Stop profiling
            profiler.disable()

            # Stop memory tracking
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Calculate runtime
            end = time.time()
            runtime = end - start

            # Output CPU profiling results
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats()
            s.seek(0)
            cpu_time_line = s.readline()  # Read the total CPU time from the first line
            print(f"Function {func.__name__} CPU time: {cpu_time_line.strip()}")

            # Output memory usage
            print(f"Memory usage for {func.__name__}:")
            print(f"Current: {current / 10**6:.6f} MB; Peak: {peak / 10**6:.6f} MB")

            # Output runtime
            print(f"Function {func.__name__} took {runtime:.6f} seconds to execute.")

        return result
    return wrapper


class PlatypusDataset:
    def __init__(self):
        self.data = load_dataset("garage-bAInd/Open-Platypus", split='train')

    def get_length(self):
        return len(self.data)

    def get_item(self, index):
        return self.data[index]

    @track_performance
    def random_sample(self, frac=0.1):
        # Random sampling using filter
        sampled_dataset = self.data.filter(lambda x: random.random() < frac)  # frac = 0.1 => ~10% sample
        return sampled_dataset
    
    @track_performance
    def stratified_sample(self, frac=0.1):
        # Stratified sampling using data_source

        # Group indices by data_source
        source_groups = defaultdict(list)
        for idx, example in enumerate(self.data):
            source_groups[example["data_source"]].append(idx)

        # Total sample size
        total_sample_size = len(self.data) * frac
        sampled_indices = []

        # Proportional sampling
        for source, indices in source_groups.items():
            proportion = len(indices) / len(self.data)
            num_samples = int(proportion * total_sample_size)
            sampled_indices.extend(random.sample(indices, min(len(indices), num_samples)))

        sampled_dataset = self.data.select(sampled_indices)
        return sampled_dataset

    @track_performance
    def clustering(self, num_clusters=3):
        # Clustering by data_source

        # Get unique data sources
        unique_sources = set(self.data["data_source"])

        # Define the number of clusters to sample
        num_clusters_to_sample = num_clusters

        # Randomly select clusters
        sampled_sources = random.sample(sorted(unique_sources), num_clusters_to_sample)

        sampled_dataset = self.data.filter(lambda x: x["data_source"] in sampled_sources)
        return sampled_dataset

    @track_performance
    def get_data_without_source(self, source_name):
        '''
        Return the subset of the dataset that doesn't include points from source_name
        '''
        return [item for item in self.data if item['data_source'] != source_name]
    
    @track_performance
    def get_data_from_source(self, source_name):
        '''
        Return the subset of the dataset from given source_name
        '''
        return [item for item in self.data if item['data_source'] == source_name]
    
    @track_performance
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

    # print(dataset.get_item(0))
    
    # Random sampling
    sampled_dataset = dataset.random_sample(0.1)
    
    # Stratified sampling
    # sampled_dataset = dataset.stratified_sample(0.1)

    # Clustering
    # sampled_dataset = dataset.clustering(3)

    # data_sources = dataset.get_sources()

    # for source in data_sources:
    #     ablated_dataset = dataset.get_data_without_source(source)
    #     source_dataset = dataset.get_data_from_source(source)
        
    #     print(len(ablated_dataset) + len(source_dataset))
        
    
    

if __name__ == "__main__":
    main()


