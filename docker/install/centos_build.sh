yum install -y wget bzip2 expect gcc which numactl-devel numatcl git devtoolset-8-toolchain centos-release-scl scl-utils-build && yum clean all && rm -rf /var/cache/yum/*
cd /home
wget --no-check-certificate -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2021.05-Linux-x86_64.sh
chmod 777 Anaconda3-2021.05-Linux-x86_64.sh

