import os

def run_grabcut(path):
    # Get a list of all file names in the specified path
    file_names = os.listdir(path)
    
    outputs_path = 'blend4'
    background_path = './data/bg'
    src_path = './data/imgs'
    background_file_names = os.listdir(background_path)
    
    # Loop over each file name
    for bg_file_name in background_file_names:
        for file_name in file_names:
            print(f"Processing file: {file_name}, bg_file_name: {bg_file_name}")
            os.makedirs(os.path.join(outputs_path, bg_file_name.split('.')[0]), exist_ok=True)
            path_src = os.path.join(src_path, file_name)
            path_tgt = os.path.join(background_path, bg_file_name)
            path_mask = os.path.join(path, file_name)
            path_out = os.path.join(outputs_path, bg_file_name.split('.')[0], file_name)
            # Check if the file is a JPEG image
            if file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
                # Construct the command string with the file name
                command = f"python poisson_blending.py --src_path {path_src} --tgt_path {path_tgt} --mask_path {path_mask} --out_path {path_out}"
                
                # Execute the command
                os.system(command)

run_grabcut('our_masks')