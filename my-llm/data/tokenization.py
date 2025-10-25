# 对数据集进行文本提取并用分词器分词
import os
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

cache_dir = "/data/huggingface_cache"
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
raw_dataset=load_dataset("rojagtap/bookcorpus", split="train",cache_dir=cache_dir)

#分批次加载数据集
def batch_iterator(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]
'''
 Dataset对象支持切片操作
 取出text字段
 yield返回一个生成器
'''   

#加载分词器
tokenizer=Tokenizer(BPE(unk_token="[UNK]"))
#预分词器
tokenizer.pre_tokenizer=Whitespace()

vocab_size=30000

#special tokens
special_tokens=["[UNK]","[CLS]","[SEP]","[PAD]","[MASK]"]

#train!
trainer=BpeTrainer(vocab_size=vocab_size,special_tokens=special_tokens)
tokenizer.train_from_iterator(batch_iterator(raw_dataset),trainer=trainer,length=len(raw_dataset))
tokenizer.save("bpe_tokenizer.json")
print("Tokenizer trained and saved as bpe_tokenizer.json")

#test
output=tokenizer.encode("Hello, how are you?")
print("Encoded output:",output.ids)
print("Decoded output:",tokenizer.decode(output.ids))
'''
text->tokenizer.encode->ids'''