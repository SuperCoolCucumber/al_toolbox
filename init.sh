python nltk_download.py
pip install jupyter_contrib_nbextensions
pip install git+https://github.com/IINemo/text_selector.git
pip install --upgrade fairscale==0.4.0
mkdir ./acleto/al4nlp/utils/packages
cd ./acleto/al4nlp/utils/packages
git clone https://github.com/Aktsvigun/summac
git clone https://github.com/neulab/BARTScore.git
cp BARTScore/bart_score.py ./
rm -rf BARTScore
rm -rf summac/.git