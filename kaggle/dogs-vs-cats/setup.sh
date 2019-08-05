current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#complete the commnon setup (install anaconda etc etc)
bash $current_dir/../setup/setup.sh
# activate conda 
source /home/ubuntu/miniconda2/bin/activate

mkdir -p $current_dir/kaggle-data
chown -R ubuntu $current_dir/kaggle-data
mkdir -p $current_dir/data
chown -R ubuntu $current_dir/data

cd kaggle-data
aws s3 sync s3://aj-kaggle/dogs-vs-cats .
unzip train.zip
unzip test1.zip
mv train ../data
mv test1 ../data

