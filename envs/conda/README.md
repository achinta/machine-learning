## Conda commands
```shell
# Create a new environment fron a conda yaml file. 
conda env create --name gpu --file=cudf13.yml
```

## Environments
### cudf13
cudf14 has an [integer overflow bug](https://github.com/rapidsai/cugraph/issues/850). Hence we go for cudf13.

If we want to install without using the environment file, here are the shell commands
```shell
conda install -y -c rapidsai -c nvidia -c numba -c conda-forge -c pytorch cudf=0.13 cugraph=0.13 numba=0.48.0 \
    cudatoolkit=10.1 pytorch torchvision scikit-learn jupyterlab jupyterlab-git nodejs
    
    
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-geometric
```

### cudf15
```shell
conda install -y -c rapidsai -c nvidia -c numba -c conda-forge -c pytorch cudf=0.15 cudatoolkit=10.1 pytorch torchvision scikit-learn jupyterlab jupyterlab-git nodejs 

pip install sklearn fastai tqdm==tqdm==4.48.2 pytorch-lightning==0.9.0 seaborn
```

### cudf16
```shell
conda install -y -c rapidsai -c nvidia -c numba -c conda-forge -c pytorch cudf=0.16 cudatoolkit pytorch torchvision scikit-learn jupyterlab jupyterlab-git nodejs 

pip install sklearn fastai tqdm==tqdm==4.48.2 pytorch-lightning==1.0.0 seaborn wandb
```

### rapdis17
Get rapids installation command from https://rapids.ai/start.html

```shell
conda create -n rapids17 -c rapidsai -c nvidia -c conda-forge \
    -c defaults rapids-blazing=0.17 python=3.8 cudatoolkit=11.0

pip install sklearn fastai tqdm==4.48.2 pytorch-lightning seaborn wandb nodejs jupyterlab jupyterlab-git
```
