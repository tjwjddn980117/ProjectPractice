echo start downloading environment for Puzzle

call conda install -y python=3.9
call conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
call conda install -y conda-forge::matplotlib conda-forge::seaborn conda-forge::tqdm anaconda::pandas

echo complete.