import pandas as pd
import os 

#get arg from command line
from sys import argv

test_name = argv[1]
file_dir = f"/home/yyyyyhc/nvdiffrast/exp/gallerytest/{test_name}/t_0_ci/"
file_path = os.path.join(file_dir, "time.csv")
save_path = f"/home/yyyyyhc/nvdiffrast/timeEval/{test_name}.csv"

df = pd.read_csv(file_path)

gf = df['gf']
intersection = df['intersection']
all_time = df['all']
op_time = all_time - intersection - gf

#get mean of the four columns
with open(save_path, 'w') as f:
    f.write("gf, intersection, op, all\n")
    f.write("{}, {}, {}, {}\n".format(gf.mean(), intersection.mean(), op_time.mean(), all_time.mean()))