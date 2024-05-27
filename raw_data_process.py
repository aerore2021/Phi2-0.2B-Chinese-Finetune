# %%
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
import pyarrow.parquet as pq
import pyarrow as pa

# %%
# 1. Wiki
origin_wiki_file = './data/wiki.txt'
tokenizer_dir = './model/tokenizer/'
liness = []
with open(origin_wiki_file, 'r', encoding='utf-8') as f_r:
    lines = f_r.readlines()
    
tokenizer  = AutoTokenizer.from_pretrained(tokenizer_dir)
ids_dtype = np.uint16
# 合并词条内容
items, content = [], []
key, k_line_idx = '', 0
content_start = False # 词条内容开始标记

for i, line in enumerate(lines):
    line_strip = line.strip()
    # 词条以冒号结尾
    if len(line_strip) > 0 and line_strip[-1] in (':', '：'):
        key = ''.join(line_strip[:-1])
        k_line_idx = i
        continue
    # 词条key在下一行，则合并上个词条
    if i == k_line_idx + 1  and key in line_strip or i == len(lines) - 1:
        txt = ''.join(content)
        if len(txt) > 0:
            items.append(txt)
        content = []
        content.append(f"{key}：")
    
    content.append(line)

def gen():
    for txt in items:
        yield {'text': txt}

dataset = Dataset.from_generator(gen, cache_dir='.cache', keep_in_memory=True)

# eos_token_id = tokenizer.eos_token_id
def txt2id_map(samples: dict, max_len: int, stride: int, tokenizer: int, ids_dtype: np.dtype, np) -> dict:
    batch_txt = samples['text']
    # eos_token_id = tokenizer.eos_token_id
    encoded = tokenizer(
        batch_txt, 
        max_length=max_len,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
        return_token_type_ids=False,
        return_offsets_mapping=False,
        return_attention_mask=False,
    )
    input_ids = encoded['input_ids']
    overflow_map = encoded['overflow_to_sample_mapping']
    
    # 获取每个doc的最后一行
    last_line_idx = []
    for idx in range(len(overflow_map) - 1):
        # 在分割处的id不一样
        if overflow_map[idx] != overflow_map[idx + 1]:
            last_line_idx.append(idx)
    # 添加最后一个doc的最后一行
    last_line_idx.append(len(overflow_map) - 1)
    # 在doc的最后一行添加eos id，如果最后一行长度为max_length，eos id覆盖最后一个token id
    # for last_idx in last_line_idx:
    #     if len(input_ids[last_idx]) == max_len:
    #         input_ids[last_idx][-1] = eos_token_id
    #     else:
    #         input_ids[last_idx] += [eos_token_id]
    
    outputs = [np.array(item, dtype=ids_dtype) for item in input_ids]
    
    return {"input_ids": outputs}

max_len, stride = 320, 0
ds = dataset.map(txt2id_map, fn_kwargs={'max_len': max_len, 'stride': stride, 'tokenizer': tokenizer, 'ids_dtype': ids_dtype, 'np': np}, batched=True, batch_size=1024, remove_columns=dataset.column_names)
ds.save_to_disk('./data/wiki')

def cut_with_end_pun(txt: str, max_len: int) -> str:
    if len(txt) <= max_len:
        return txt
    i = max_len
    while i >= 0 and txt[i] not in ('。', '！'):
        i -= 1
        
    end = max_len if i <= 0 else i + 1
    txt = ''.join(txt[0: end])
    return txt

def corpus2chunk(texts: list[str], batch_sz: int=512^2, max_len: int=320, window_size: int = 2)-> list[str]:
    buffer, buffer_len = [], 0
    chunk_data = []
    for i, line in enumerate(texts):
        buffer_len += len(line)
        buffer.append(line)
        
        if buffer_len >= batch_sz or i == len(texts) - 1:
            buffer_txt = ''.join(buffer)
            for i in range(0, len(buffer_txt), max_len - window_size):
                chunk_data.append(''.join(buffer_txt[i:i+max_len]))
            buffer, buffer_len = [], 0
            
    return chunk_data

chunk_data = corpus2chunk(items)
tb = pa.Table.from_arrays([chunk_data], names=['text'])
pq.write_table(table=tb, where='./data/wiki_chunk.parquet', row_group_size=50000, data_page_size=50000, )

# %%
# 2. 