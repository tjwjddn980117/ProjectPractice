@echo off

rem 가상 환경 이름
set ENV_NAME=Puzzle

rem 가상 환경 생성 및 활성화
conda create -y -n %ENV_NAME% python=3.9
call activate %ENV_NAME%

rem 필수 패키지 설치
conda install -y python=3.9
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y matplotlib seaborn tqdm pandas

rem 가상 환경 비활성화
call conda deactivate

echo complete.