FROM kaggle/python-gpu-build

ENV lang="ja_jp.utf-8" language="ja_jp:ja" lc_all="ja_jp.utf-8"

ADD . /tmp/working
WORKDIR /tmp/working
RUN pip install -r requirements.txt