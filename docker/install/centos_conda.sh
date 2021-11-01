cd /home
bash Anaconda3-2021.05-Linux-x86_64.sh -b -p /usr/local/anaconda3
rm Anaconda3-2021.05-Linux-x86_64.sh
echo "export PATH="/usr/local/anaconda3/bin:$PATH"" > ~/.bashrc
source ~/.bashrc

