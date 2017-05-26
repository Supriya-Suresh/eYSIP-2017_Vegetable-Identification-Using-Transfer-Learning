import requests
import os
from shutil import copy2
wnetids = {"cauliflower":'07715103',"beetroot":'07719839',"brinjal":'07713074',"parsley":'07819896',"ladyfinger":'07733394'}

for name in wnetids:
    s = './' + name
    if not os.path.exists(s):
        os.makedirs(s)
    url = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n" + wnetids[name]
    print(url)
    data = requests.get(url)
    filepath = s + '/' + name + '.txt'
    with open(filepath,'w') as f:
        f.write(data.text)
    copy2('./fetch.py',s)
    