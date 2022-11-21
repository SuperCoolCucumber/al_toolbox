rsync -av --progress ../ ./acleto --exclude docker --exclude .git
docker build -t acleto_demo_cuda113 -f Dockerfile_cuda113 ./
