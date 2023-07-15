sudo apt-get update
sudo apt-get install -y zip build-essential
wget -L https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh
bash Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -b -p ~/miniconda
rm Miniconda3-py310_23.5.2-0-Linux-x86_64.sh
echo 'export PATH="~/miniconda/bin:$PATH"' >> ~/.bashrc 
source ~/.bashrc
conda update conda -y
conda create -p ~/venv_p310/ python=3.10 -y
source activate ~/venv_p310/
pip install numpy numba scipy matplotlib notebook tqdm
