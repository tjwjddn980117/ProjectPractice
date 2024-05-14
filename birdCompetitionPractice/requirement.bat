rem you should change the root with your own environment path root.
rem and you could change the ENV_NAME with your one vitual environment.
set root=C:\Users\qowor\anaconda3
set ENV_NAME=Bird

if not exist "%root%" (
    echo check the root please: %root%
    pause
    exit
)

call %root%\Scripts\activate.bat %root%

echo make the virtual environment '%ENV_NAME%'
call conda create -y -n %ENV_NAME% python=3.10

echo enter the virtual environment.
call conda activate %ENV_NAME%

echo start downloading environment for %ENV_NAME%.
call conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
call conda install -y anaconda::pandas anaconda::scikit-learn anaconda::scipy conda-forge::torchinfo
pip install --quiet timm pytorch_lightning==1.7.7 torchmetrics==0.11.1
pip install opencv-python
pip install transformers
pip install wandb

call conda deactivate

echo complete. 