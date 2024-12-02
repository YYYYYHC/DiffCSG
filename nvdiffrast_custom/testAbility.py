import os
import pdb
failure_count = 0
success_count = 0
def find_last_loss_number(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == 'log.txt':
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in reversed(lines):
                        if 'loss:' in line:
                            global failure_count
                            global success_count
                            loss_number = line.split('loss_true:')[-1].strip()
                            if float(loss_number) > 5e-4:
                                print(f"Loss number: {loss_number}, File path: {file_path}")
                                failure_count += 1
                            else:
                                success_count += 1    
                            break
                            

# Usage
folder_path = '/home/cli7/CSGDR/nvdiffrast/testAbility'
find_last_loss_number(folder_path)
print(f"Failure count: {failure_count}, Success count: {success_count}")
