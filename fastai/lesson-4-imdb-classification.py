
import fastai
from fastai import *
from fastai.text import *
from fastai.utils.mod_display import *
from datetime import datetime


from fastprogress import force_console_behavior
import fastprogress
fastprogress.fastprogress.NO_BAR = True
master_bar, progress_bar = force_console_behavior()
fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar
fastprogress.fastprogress.WRITER_FN = str

path = untar_data(URLs.IMDB)

# ### Using the datablock api


bs = 48
data_lm = TextList.from_folder(path)\
            .filter_by_folder(include=['train','test'])\
            .split_by_rand_pct(0.1)\
            .label_for_lm()\
            .databunch(bs=bs)

learn = language_model_learner(data_lm,AWD_LSTM,drop_mult=0.3)

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Training Started')
# with progress_disabled_ctx(learn) as learn:
learn.fit_one_cycle(1,1e-2,moms=(0.8,0.7))

learn.unfreeze()
learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))
learn.save_encoder('enc')
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Training complete')