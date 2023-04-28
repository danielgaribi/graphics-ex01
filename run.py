import os

def run_grabcut(path):
    # Get a list of all file names in the specified path
    file_names = os.listdir(path)
    
    outputs_path = 'outputs6'
    # Loop over each file name
    for n_comp in [5, 2, 8]:
        for k_blur in [1, 7, 15]:
            for file_name in file_names:
                print(f"Processing file: {file_name}, n_comp: {n_comp}, k_blur: {k_blur}")
                os.makedirs(os.path.join(outputs_path, f'n_comp_{n_comp}_k_blur_{k_blur}', file_name.split('.')[0]), exist_ok=True)
                # Check if the file is a JPEG image
                if file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
                    # Construct the command string with the file name
                    command = f"python grabcut.py --input_name {file_name.split('.')[0]} --n_comp {n_comp} --k_blur {k_blur} --out_path {os.path.join(outputs_path, f'n_comp_{n_comp}_k_blur_{k_blur}')}"
                    
                    # Execute the command
                    os.system(command)

run_grabcut('data/imgs')