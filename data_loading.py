import numpy as np
import pgn_interpreter
import os
from threading import Thread, Lock

def process_condensed_pgn(source_dir, file, target_dir, lock):
    # Load the PGNs from a condensed PGN file, convert them to numpy arrays, and save them as npz files
    # This is preprocessing, we don't need to do this every time we run the program
    # This function will be run on multiple threads,
    # First, make sure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if file.endswith(".pgn"):
        # pgn_interpreter has a generator called read_condensed_pgn_to_bitboard_and_policy
        # It reads a PGN file and returns a generator that yields bitboards, policies, and results
        # The generator returns boards 20000 at a time
        # We want to save each batch of boards to a separate file
        # When the generator is exhausted, we will have saved all the boards to npz files
        generator = pgn_interpreter.read_condensed_pgn_to_bitboard_and_policy(os.path.join(source_dir, file))
        batch_num = 0
        # Loop until the generator is exhausted - the next file will need its own generator
        while True:
            try:
                # Get the next batch of boards
                bitboards, policies, results = next(generator)
                # Save the batch to a file
                lock.acquire()
                np.savez_compressed(os.path.join(target_dir, f"{file.split('.')[0]}_{batch_num}"), bitboards=bitboards, policies=policies, results=results)
                lock.release()
                batch_num += 1
            except StopIteration:
                break

def process_files(source_dir, files, target_dir, lock):
    """Helper function to process a batch of files within a thread"""
    for file in files:
        if check_if_processed(target_dir, file):
            print(f"Skipping {file} - already processed")
        else:
            process_condensed_pgn(source_dir, file, target_dir, lock)

def process_source_dir(source_dir, target_dir, num_threads=6):
    """
    Process all PGN files in the source directory using multiple threads
    
    Args:
        source_dir: Directory containing PGN files to process
        target_dir: Directory where processed numpy files will be saved
        num_threads: Number of threads to use for parallel processing
    """
    # Get all PGN files in the source directory
    pgn_files = [f for f in os.listdir(source_dir) if f.endswith('.pgn')]
    
    if not pgn_files:
        print(f"No PGN files found in {source_dir}")
        return
    
    # Create a shared lock for file operations
    lock = Lock()
    
    # Create and start threads
    threads = []
    for i in range(num_threads):
        # Distribute files evenly across threads (every nth file goes to thread i)
        thread_files = pgn_files[i::num_threads]
        if thread_files:  # Only create thread if there are files to process
            thread = Thread(target=process_files, args=(source_dir, thread_files, target_dir, lock))
            threads.append(thread)
            thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print(f"Processed {len(pgn_files)} PGN files from {source_dir} to {target_dir}")

def check_if_processed(target_dir,file):
    # Kind of hacky - files get processed into batches, but most of the time it's either 26 or 27 batches
    # We're just going to see if the 25th batch has been processed - if so, we'll assume everything has

    target_file_name = file.split('.')[0] + '_25.npz'
    return os.path.exists(os.path.join(target_dir, target_file_name))

def verify(dir):
    # Load each file in the directory and check that number of boards, policies, and results match
    # Make sure to dump each file before loading the next one to avoid memory issues

    for file in os.listdir(dir):
        if file.endswith('.npz'):
            data = np.load(os.path.join(dir, file))
            bitboards = data['bitboards']
            policies = data['policies']
            results = data['results']
            data.close()
            assert bitboards.shape[0] == policies.shape[0] == results.shape[0], f"Shapes do not match for {file}"
            print(bitboards[0].shape)
            print(policies[0].shape)

if __name__ == '__main__':
    verify('data/processed-pgns')