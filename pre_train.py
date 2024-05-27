import os
import platform
import time
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datasets import Dataset, load_dataset
import torch
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling, PhiConfig, PhiForCausalLM, TrainerCallback, TrainingArguments, TrainerState, TrainerControl, Trainer 

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
attn_impl = 'flash_attention_2'
try:
    from flash_attn import flash_attn_func
except Exception as e:
    attn_impl = 'eager'

# %%
# 1. 训练数据和设定
TRAIN_FILES = [
    
]

EVAL_FILE = ''
# dataclass 能够自动添加特殊方法到用户定义的类中
@dataclass
class PretrainArguments:
    tokenizer_dir: str = './model_save/tokenizer/'
    model_save_dir: str = './model_save/pre/'
    logs_dir: str = './logs/'
    # default_factory接受一个无参数的函数，默认是TRAIN_FILES
    train_files: list[str] = field(default_factory=lambda: TRAIN_FILES)
    eval_file: str = EVAL_FILE
    max_seq_len: int = 512
    attn_impl: str = 'eager' if platform.system() == 'Windows' else attn_impl

pretrain_args = PretrainArguments()


# %%
# 2. 加载训练好的tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrain_args.tokenizer_dir)
vocab_sz = len(tokenizer)
if vocab_sz % 64 != 0:
    vocab_sz = (vocab_sz // 64 + 1) * 64
print(f"source vocab size: {len(tokenizer)}, final vocab size: {vocab_sz}")
# token to id缓存到文件，使用的时候不用再次tokenize
map_dtype = np.uint16

def token2id(samples: dict[str, list]) -> dict:
    batch_text = samples['text']
    # 不截断，不填充，不返回attention mask
    outputs = tokenizer(
        batch_text,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )
    # 返回文本中每个token对应的标识符
    input_ids = [np.array(item, dtype=map_dtype) for item in outputs["input_ids"]]
    return {"input_ids": input_ids}

# %%
# 3. 加载数据集
def get_maped_dataset(files: str|list[str]) -> Dataset:
    dataset = load_dataset(path='parquet', data_files=files, split='train', cache_dir='.cache')
    maped_dataset = dataset.map(token2id, batched=True, batch_size=1024, remove_columns=dataset.column_names)
    return maped_dataset

train_dataset = get_maped_dataset(pretrain_args.train_files)
eval_dataset = get_maped_dataset(pretrain_args.eval_file)
print(train_dataset, eval_dataset)

# %%
# 4. model config
# 训练clm模型，能够更好地生成流畅文本
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
# flash_attention_2需要set_default_dtype=bfloat16
if pretrain_args.attn_impl == 'flash_attention_2':
    torch.set_default_dtype(torch.bfloat16)
    
phi2_config = PhiConfig(
    vocab_size=vocab_sz,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    hidden_size=960,
    num_attention_heads=24,
    max_position_embeddings=512,
    intermediate_size=4096,
    attn_implementation=pretrain_args.attn_impl,
)

model = PhiForCausalLM(phi2_config)
model_size = sum(t.numel() for t in model.parameters())
print(f"Phi2 size: {model_size / 1000**2:.1f}M parameters.")

# %%
# 5. cuda cache callback
# callback 主要用于训练过程中地实时保存
class MYTrainerCallback(TrainerCallback):
    log_cnt = 0
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # 打印n次日志后清除cuda缓存
        self.log_cnt += 1
        if self.log_cnt % 2 == 0:
            torch.cuda.empty_cache()
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # 保存一次模型
        control.should_save = True
        return control

my_trainer_callback = MYTrainerCallback()

# %%
# 6. Training config
args = TrainingArguments(
    output_dir=pretrain_args.model_save_dir,
    per_device_train_batch_size=5,
    gradient_accumulation_steps=32,
    num_train_epochs=4,
    weight_decay=0.1,
    warmup_steps=1000,
    learning_rate=5e-4,
    evaluation_strategy='steps',
    eval_steps=2000,
    save_steps=2000,
    save_strategy='steps',
    save_total_limit=3,
    report_to='tensorboard',
    optim='adafactor',
    bf16=True,
    logging_steps=5,
    log_level='info',
    logging_first_step=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[my_trainer_callback],
)
# %%
# 7. 开始训练
trainer.train()
# 计算perplexity
eval_resuls = trainer.evaluate()
print(f"Perplexity: {np.exp(eval_resuls['eval_loss']):.2f}")

# %%
# 8. 最后保存训练的loss日志和模型
loss_log = pd.DataFrame(trainer.state.log_history)
loss_log.to_csv(f"./logs/pre_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")
trainer.save_model(pretrain_args.model_save_dir)