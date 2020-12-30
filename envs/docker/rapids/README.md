### rapids
added torch and tensorflow to rapids container. 

```shell
# build
docker-compose build

# run
docker-compose run -d --name rapids -p 8888:8888 -p 8787:8787 -p 8786:8786 -v ~/machine-learning:/rapids/ml rapids bash -c 'source ~/.bashrc;jupyter notebook --allow-root --ip 0.0.0.0'

# check logs for token
docker logs rapids

```
