/home/ubuntu/Miniconda2-latest-Linux-x86_64.sh:
  cmd.run:
    - name: wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
    - cwd: /home/ubuntu
    - creates: /home/ubuntu/miniconda2

/home/ubuntu/:
  cmd.script:
    - name: /home/ubuntu//Miniconda2-latest-Linux-x86_64.sh
    - args: -b -p /home/ubuntu/miniconda2
    - creates: /home/ubuntu/miniconda2

#change the permissions
/home/ubuntu/miniconda2:
  file.directory:
    - user: ubuntu
    - group: ubuntu
    - mode: 755
    - makedirs: True
    - recurse:
      - user
      - group
      - mode

# /home/ubuntu/.bashrc:
#   file.blockreplace:
#     - filename: /home/ubuntu/.bashrc
#     - marker_start: "# START managed zone by saltstack. DO-NOT-EDIT"
#     - marker_end: "# END managed zone by saltstack"
#     - append_if_not_found: True
#     - show_changes: True
#     - backup: '.bak'
#     - content: 'export PATH=/home/ubuntu/miniconda2/bin:$PATH'

install_conda_modules:
  cmd.script:
    - name: salt://install_conda_modules.sh
    - user: ubuntu

