# DiffCSG

This is the official implementation of our Siggrapha Asia 2024 paper "DiffCSG: Differentiable CSG via Rasterization".

Our Implementation are based on [Nvdiffrast](https://github.com/NVlabs/nvdiffrast).

## ToDo

- [x] Release Code and Data (Will be done before 2025, I promise.)
- [ ] Clean the code and usage (Very soon before Jan 2025.)
- [ ] Upgrade for better performance .

## Usage

Install nvdiffrast under nvdiffrast_custom

Then run the benchmark under ./nvidffrast_custom:

        sudo ./run_sample.sh --build-container  ./samples/torch/CSGDR.py  --use_random --scale_level 0 --USE_ALL_PARAM
        

## Contribute to this project

If you are familiar with cuda, openGL and Nvdiffrast, feel free to contact me at H.C.Yuan@ed.ac.uk 

