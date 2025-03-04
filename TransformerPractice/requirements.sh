# 가상 환경 이름
ENV_NAME="Transformer"

# 가상 환경 생성 및 활성화
conda create -n $ENV_NAME python=3.8
conda activate $ENV_NAME

# 필수 패키지 설치
conda install python==3.8
conda install -y pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
conda install -y pytorch::torchtext conda-forge::spacy

# 가상 환경 비활성화
deactivate

echo "complete."