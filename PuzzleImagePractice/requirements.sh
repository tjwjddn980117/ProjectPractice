# 가상 환경 이름
ENV_NAME="Puzzle"

# 가상 환경 생성 및 활성화
conda create -n $ENV_NAME python=3.9
conda activate $ENV_NAME

# 필수 패키지 설치
conda install python==3.9
conda install conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y conda-forge::matplotlib conda-forge::seaborn conda-forge::tqdm anaconda::pandas

# 가상 환경 비활성화
deactivate

echo "complete."