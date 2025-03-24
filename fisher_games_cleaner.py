import os

def clean_and_condense_PGNs(src, dest, final):
    # Create output directories if they don't exist
    if not os.path.exists(dest):
        os.makedirs(dest)
    if not os.path.exists(final):
        os.makedirs(final)

    print("Cleaning PGNs...")
    files = os.scandir(src)
    count_standard = 0
    count_fischer = 0
    condensing_count_standard = 1
    condensing_count_fischer = 1
    
    # Open output files
    current_file_standard = open(os.path.join(final, f'condensed_standard{condensing_count_standard}.pgn'), 'w+')
    current_file_fischer = open(os.path.join(dest, f'condensed_fischer{condensing_count_fischer}.pgn'), 'w+')
    
    file_count = 0
    try:
        for file in files:
            file_count += 1
            if file_count % 10000 == 0:
                print(f"Processed {file_count} files...")
                
            if file.name.endswith('.pgn'):
                try:
                    with open(file.path, 'r') as f:
                        # Read first line to check if it's Fischer Random
                        first_line = f.readline()
                        
                        if first_line.startswith('[FEN "'):
                            # Don't forget to write the first line
                            current_file_fischer.write(first_line)
                            # Write the rest of the file
                            for line in f:
                                current_file_fischer.write(line)
                            current_file_fischer.write("\n\n")  # Add separator between games
                            
                            count_fischer += 1
                            if count_fischer % 5000 == 0:
                                current_file_fischer.close()
                                condensing_count_fischer += 1
                                # Fix: Use correct directory (dest) for Fischer files
                                current_file_fischer = open(os.path.join(dest, f'condensed_fischer{condensing_count_fischer}.pgn'), 'w+')
                        else:
                            # Don't forget to write the first line
                            current_file_standard.write(first_line)
                            # Write the rest of the file
                            for line in f:
                                current_file_standard.write(line)
                            current_file_standard.write("\n\n")  # Add separator between games
                            
                            count_standard += 1
                            if count_standard % 5000 == 0:
                                current_file_standard.close()
                                condensing_count_standard += 1
                                current_file_standard = open(os.path.join(final, f'condensed_standard{condensing_count_standard}.pgn'), 'w+')
                except Exception as e:
                    print(f"Error processing file {file.name}: {e}")
    except Exception as e:
        print(f"Error during file processing: {e}")
    finally:
        # Ensure files are closed even if an error occurs
        current_file_standard.close()
        current_file_fischer.close()
    
    print(f"Cleaning complete. Processed {count_fischer} Fischer games and {count_standard} standard games.")

if __name__ == '__main__':
    src = 'pgns-run1'
    dest = 'pgns-run1-fischer'
    final = 'condensed-pgns'

    clean_and_condense_PGNs(src, dest, final)