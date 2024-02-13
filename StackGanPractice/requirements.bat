echo start downloading environment for StackGan

call conda install -y python=3.7
call conda install -y pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
call conda install -y conda-forge::torchfile conda-forge::easydict conda-forge::python-dateutil conda-forge::tensorboard anaconda::pandas

echo complete.