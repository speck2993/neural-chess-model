def fix_name(name):
    parts = name.split('.')
    n = parts[1].split('_')[1]
    return parts[0] + '_' + n + '.' + parts[2]

def clean_file_names(dir):
    # Files have bad name formats. Should be {source_file}_{n}.npz, but is {source_file}.pgn_{n}.npz instead
    # We need to rename them
    import os
    for file in os.listdir(dir):
        if file.endswith('.npz'):
            new_name = fix_name(file)
            os.rename(dir + '/' + file, dir + '/' + new_name)

if __name__ == '__main__':
    clean_file_names('data/processed-pgns')