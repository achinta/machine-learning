### rapids
added torch and tensorflow to rapids container. 

```shell
# build
docker-compose build

# run. Not using docker-compose as I am not sure how to pass the --gpus flag in compose
docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 -v ~/machine-learning:/rapids/ml rapids17 bash -c 'source ~/.bashrc;jupyter notebook --allow-root --ip 0.0.0.0'

# check logs for token
docker logs rapids

```
