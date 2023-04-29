import os
import csv

# Define the directory where the input text files are located
input_dir = r'C:\Users\tlust\Desktop\git\graphics-ex01\outputs2'

# Open the output CSV file
with open('output.csv', 'w', newline='') as outfile:

    # Create a CSV writer object and write the header row
    writer = csv.writer(outfile)
    writer.writerow(['k_blur', 'n_comp', 'img_name', 'Accuracy', 'Jaccard', 'rect', 'run_time', 'nof_iter'])

    # Loop over each subdirectory in the input directory
    for root, dirs, files in os.walk(input_dir):
        # Loop over each file in the subdirectory
        for file in files:
            # Check if the file is a text file
            if file == 'metrics.txt':
                # Get the full path of the text file
                txt_file = os.path.join(root, file)
                # Open the text file
                with open(txt_file, 'r') as infile:
                    # Initialize the variables
                    k_blur = 0
                    n_comp = 0
                    img_name = ''
                    accuracy = 0.0
                    jaccard = 0.0
                    rect = ''
                    run_time = ''
                    nof_iter = 0
                    # Loop over each line in the text file
                    for line in infile:
                        # Split the line into fields
                        if line.startswith('input_name:'):
                            img_name = line.split(':')[1].strip()
                        elif line.startswith('blur:'):
                            k_blur = int(line.split(':')[1].strip())
                        elif line.startswith('n_copm:'):
                            n_comp = int(line.split(':')[1].strip())
                        elif line.startswith('Accuracy='):
                            accuracy = line.split('=')[1].strip().split(',')[0]
                            jaccard = line.split('=')[2].strip().split(',')[0]
                            accuracy = float(accuracy.strip())
                            jaccard = float(jaccard.strip())
                        elif line.startswith('rect:'):
                            rect = line.split(':')[1].strip()
                        elif line.startswith('run_time:'):
                            run_time = line.split(':')[1].strip()
                        elif line.startswith('nof_iters:'):
                            nof_iter = int(line.split(':')[1].strip())
                    # Write the row to the CSV file
                    writer.writerow([k_blur, n_comp, img_name, accuracy, jaccard, rect, run_time, nof_iter])
