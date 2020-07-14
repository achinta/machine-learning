## Conda commands
```shell
# Create a new environment fron a conda yaml file. 
conda env create --name gpu --file=cudf13.yml
```

## Environments
### cudf13
cudf14 has an [integer overflow bug](https://github.com/rapidsai/cugraph/issues/850). Hence we go for cudf13
