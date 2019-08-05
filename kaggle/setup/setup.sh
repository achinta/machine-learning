#install salt
current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
apt-get -y install salt-minion
apt-get -y install unzip
mkdir -p /srv/salt

#install python dependencies
cp $current_dir/install_conda_modules.sh /srv/salt

#install anaconda
cp $current_dir/anaconda.sls /srv/salt
sudo salt-call state.sls --local anaconda

mkdir -p /home/ubuntu/.jupyter
cp jupyter_notebook_config.py /home/ubuntu/.jupyter/jupyter_notebook_config.py
chown -R ubuntu ~/.jupyter 


