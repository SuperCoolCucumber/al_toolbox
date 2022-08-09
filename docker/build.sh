rsync -av --progress ../ ./acleto --exclude docker --exclude .git
docker build -t acleto_demo ./
