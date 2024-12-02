SCALE={'0':0.05, '1':0.1, '2':0.2} #[-0.1,+0.1], [-0.2,+0.2], [-0.4,+0.4]
BASE_PATH = '/home/yyyyyhc/nvdiffrast/CodeBase/benchmark' #replace with your own path
import os
import torch
import json
def pertub(code_source='CADTalk', code_name='bike', scale_level='0'):
    param_file = os.path.join(BASE_PATH, code_source, code_name, 'source.param')
    config_path = os.path.join(BASE_PATH, code_source, code_name, 'config.json')
    if not os.path.exists(config_path):
        print('No such file:', config_path)
        ignore = []
    else:
        config = json.load(open(config_path))
        ignore = config['param_ignore']
        
    variables = {}
    with open(param_file, 'r') as f:
        for line in f:
            name, value = line.strip().split('=')
            
            variables[name.strip()] = torch.tensor(float(value.strip()),dtype=torch.float32).cuda()
            if name.strip() == 's' or name.strip() == 'scale': #we don't optimize global scaling
                continue                  
    
    variables_count = len(variables)
    size = (variables_count,)
    random_numbers = torch.torch.normal(0, SCALE[scale_level], size=size)
    print(random_numbers)
    target_variables = {}
    for i,key in enumerate(variables):
        random_number = random_numbers[i]
        if key in ignore:
            random_number = 0
        target_variables[key] = variables[key].clone()*(1 +random_number)
    print(target_variables)
    
if __name__ == '__main__':
    pertub('CADTalk', 'bike', '0')
    