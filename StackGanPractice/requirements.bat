rem you should change the root with your own environment path root.
rem and you could change the ENV_NAME with your one vitual environment.
set root=C:\Users\qowor\anaconda3
set ENV_NAME=StackGan

call %root%\Scripts\activate.bat %root%

echo make the virtual environment '%ENV_NAME%'
call conda create -y -n %ENV_NAME% python=3.9

echo enter the virtual environment.
call conda activate %ENV_NAME%

echo start downloading environment for %ENV_NAME%.
call conda install -y pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
call conda install -y conda-forge::torchfile conda-forge::easydict conda-forge::python-dateutil conda-forge::tensorboard anaconda::pandas

call conda deactivate

echo complete.