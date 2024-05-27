# %%
from os import sep
import tokenizers
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
import tokenizers.normalizers
import tokenizers.pre_tokenizers
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Punctuation, Digits, Metaspace
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

# %%
# 1. Corpus
corpus_file = './data/wiki.txt'
tokenizer_save_dir = './model_save/hf_bpe_tokenizer.json'

# %%
# 2. Train tokenizer Config
def train_tokenizer(max_train_line: int=None, token_type: str='char') -> None:
    def get_training_corpus(buffer_size: int=1000, chunk_len: int=2048) -> list:
        line_cnt = 0
        buffer = []
        with open(corpus_file, 'r', encoding='utf-8') as f_r:
            cur_chunk_txt, txt_len = [], 0
            for line in f_r:
                cur_chunk_txt.append(line)
                txt_len += len(line)
                line_cnt += 1
                if txt_len >= chunk_len:
                    buffer.append(
                        ''.join(cur_chunk_txt)
                    )
                    cur_chunk_txt, txt_len = [], 0
                if len(buffer) >= buffer_size:
                    yield buffer
                    buffer = []
                if isinstance(max_train_line, int) and line_cnt >= max_train_line:
                    break
            if len(buffer) > 0: yield buffer

    special_tokens = ["[PAD]","[EOS]","[SEP]","[BOS]", "[CLS]", "[MASK]", "[UNK]"]

    if token_type == 'char':
        model = BPE(unk_token="[UNK]")
        tokenizer = Tokenizer(model)
        # 用兼容等价分解合并对utf编码进行等价组合，比如全角A to 半角A 
        tokenizer.normalizer = tokenizers.normalizers.Sequence([NFKC()])
        # 标点符号，数字和Metaspace预分割
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence([Punctuation(), Digits(individual_digits=True), Metaspace()])
        tokenizer.add_special_tokens(special_tokens)
        tokenizer.decoder = decoders.Metaspace()
    elif token_type == 'byte':
        # no need of unk_token
        model = BPE()
        tokenizer = Tokenizer(model)
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tokenizer.add_special_tokens(special_tokens)
        tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False, use_regex=True)
        tokenizer.post_processor = tokenizers.processors.ByteLevel(trim_offsets=False)
    else:
        raise Exception('Token type must be `char` or `byte`')
    
    trainer = BpeTrainer(vocab_size=40960, min_frequency=100, show_progress=True, special_tokens=special_tokens)
    
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    
    if '\t' not in tokenizer.get_vocab():
        tokenizer.add_tokens(['\t'])
    if '\n' not in tokenizer.get_vocab():
        tokenizer.add_tokens(['\n'])
    tokenizer.save(tokenizer_save_dir)
    
# %%
# 4. Train a tokenizer
train_tokenizer(token_type='char')

# %%
# 5. 将tokenizer转换为transformers的tokenizer
slow_tokenizer = Tokenizer.from_file(tokenizer_save_dir)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=slow_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    mask_token="[MASK]",
    bos_token="[BOS]",
    eos_token="[EOS]",
    cls_token="[CLS]",
    sep_token="[SEP]",
)
tokenizer.save_pretrained('./model_save/')