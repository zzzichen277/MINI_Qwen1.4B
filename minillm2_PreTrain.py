#https://github.com/jiahe7ay/MINI_LLM/blob/main/pre_train.py修改版
from dataclasses import dataclass, field
import platform,time,os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerControl, TrainerState

from datasets import Dataset, load_dataset
from qwen.configuration_qwen import QWenConfig
from qwen.modeling_qwen import QWenLMHeadModel

from qwen.tokenization_qwen import QWenTokenizer
# torch._dynamo.config.optimize_ddp = False
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

attn_implementation = "flash_attention_2"
try:
    from flash_attn import flash_attn_func
except Exception as e:
    attn_implementation = "eager"

#训练数据来源 这里以中文wiki和中文baike为例 需添加sky150b bell
TRAIN_FILES = [
     './datasets_Processed/wikipedia_filter.parquet',
     './datasets_Processed/baike_chunk_512_5.6M_0.parquet',
     './datasets_Processed/baike_chunk_512_5.6M_1.parquet',
]
EVAL_FILE = "./datasets_Processed/pretrain_eval_512_1w.parquet"
#===================================================================================
# 定义pretrain训练配置
@dataclass
class PretrainConfig:
    max_seq_len: int = 512                                          #最大序列长度,表示模型输入序列的最大长度限制。
    
    model_dir: str = "./qwen/"                                      #模型目录,表示model模型文件的存储路径。
    tokenizer_dir: str = "./qwen/"                                  #分词器目录,表示分词器模型文件的存储路径。                 
    model_save_dir: str = "./model_save/train/"                     #模型保存目录,表示预训练模型文件的存储路径。
    logs_dir: str = "./logs/"                                       #日志目录,表示日志文件的存储路径。   

    train_files: list = field(default_factory=lambda: TRAIN_FILES)  #训练文件列表，表示预训练数据的文件列表。
    eval_file: str = EVAL_FILE                                      #评估文件，表示评估数据的文件路径。
    cache_dir = ".cache"
    attn_implementation: str = ( "eager" if platform.system() == "Windows" else attn_implementation) # Windows 使用默认的attention实现，    



#自定义的Trainer回调类，可以在训练过程中的不同阶段执行自定义的操作。
class MyTrainerCallback(TrainerCallback):
    log_cnt = 0

    def on_log(self,args: TrainingArguments,
        state: TrainerState, 
        control: TrainerControl, **kwargs,
        ):
        """ 在打印 n 次日志后清除cuda缓存，适合低显存设备，能防止OOM"""
        self.log_cnt += 1
        if self.log_cnt % 2 == 0:
            torch.cuda.empty_cache()

    def on_epoch_end(self,args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl, **kwargs,
        ):
       
        """在on_epoch_end时保存一次模型  """
        # 设置should_save=True并返回即可
        control.should_save = True
        return control

my_trainer_callback = MyTrainerCallback()

# 定义pretrain函数
def train_pretrain(config: PretrainConfig,) -> None:
    
    #step1. 加载训练好的tokenizer
    tokenizer = QWenTokenizer.from_pretrained(config.tokenizer_dir)
    tokenizer.pad_token_id = tokenizer.im_end_id  

    vocab_size = len(tokenizer)     #词表大小设置64的整数倍
    if vocab_size % 64 != 0:
        vocab_size = (vocab_size // 64 + 1) * 64
    # 词表小于 65535用uint16存储，节省磁盘空间，否则用uint32存储
    print(f"final vocab size: {vocab_size}")
    map_dtype = np.uint16 if vocab_size < 65535 else np.uint32

    # token to id缓存到文件，使用时不用再次tokenize
    def token_to_id(samples: dict) -> dict:
    
        batch_txt = samples["text"]
        outputs = tokenizer(
            batch_txt,truncation=False,
            padding=False,return_attention_mask=False,
        )
        input_ids = [np.array(item[0:512], dtype=map_dtype) for item in outputs["input_ids"]]  #截断，防止padding太大
        return {"input_ids": input_ids}

    #加载数据集转换模型所需的格式
    def get_maped_dataset(split: str, files: str,
        cache_dir: str = '.cache') -> Dataset:
    
        dataset = load_dataset(path="parquet",data_files=files, split=split,
            cache_dir=cache_dir ,  keep_in_memory=False,  
        ) 
       
        maped_dataset = dataset.map(token_to_id,  batched=True,batch_size=10000,
            remove_columns=dataset.column_names,num_proc=24,keep_in_memory=False,
        )     
        ###返回映射后的数据集
        return maped_dataset           

    

    #step2.定义qwen模型-读取config.json配置参数
    configs = QWenConfig.from_pretrained(config.model_dir)
    model = QWenLMHeadModel(configs)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"QWen size: {model_size / 1000**2:.1f}M parameters")

    #step3.加载预训练数据集
    train_dataset = get_maped_dataset(split="train",files=config.train_files)
    eval_dataset = get_maped_dataset(split="train",files=config.eval_file)
    print(train_dataset, eval_dataset)

    #step4.定义data_collator，用于准备模型训练所需的输入数据，但不进行掩码语言建模任务。
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    #如果配置flash_attention_2，需手动设置set_default_dtype为float16
    #if args.attn_implementation == "flash_attention_2":
    #    torch.set_default_dtype(torch.bfloat16)

    #step5.定义qwen预训练参数
    args = TrainingArguments(
        output_dir=config.model_save_dir,          #指定模型训练过程中输出文件的保存路径
        per_device_train_batch_size=8,             #每个训练设备的训练批量大小
        per_device_eval_batch_size=4,              #每个评估设备的评估批量大小。
        gradient_accumulation_steps=10,            #梯度累积步数，用于将多个小批量的梯度累积后再进行参数更新。
        num_train_epochs=1,                        #训练的 epoch 数量，即遍历整个训练数据集的次数。
        weight_decay=0.1,                          #权重衰减（L2 正则化）的系数，用于控制模型参数的大小。
        ddp_find_unused_parameters=False,          #是否在分布式训练中查找未使用的参数。
        warmup_steps=0,                            #学习率预热步数，即在训练开始阶段逐渐增加学习率的步数。   
        learning_rate=1e-4,                        #初始学习率。
        evaluation_strategy="steps",               #评估策略，可以是 "no"、"steps"或者 "epoch"。
        eval_steps=100,                            #评估步数，用于指定在 evaluation_strategy 为 "steps" 时每次评估的步数。
        save_steps=500,                            #保存模型的步数间隔，即每经过 save_steps 步训练后保存一次模型。
        save_strategy="steps",                     #保存策略，可以是 "steps"或者 "epoch"
        save_total_limit=4,                       #保存模型的数量限制，即保存模型文件的最大数量。
        report_to="tensorboard",                  #指定报告输出的目标，例如 "tensorboard" 表示输出到 TensorBoard。
        optim="adamw_torch",                      #优化器的类型，例如 "adamw_torch" 表示使用 PyTorch 中的 AdamW 优化器。
        lr_scheduler_type="cosine",               #学习率调度器的类型，例如 "cosine" 表示使用余弦退火学习率调度器。
        bf16=True,                                #是否启用 bfloat16 混合精度训练。
        logging_steps=20,                         #记录日志的步数间隔。 
        log_level="info",                         #日志级别，例如 "info" 表示记录信息级别的日志。
        logging_first_step=True,                  #是否记录训练的第一步日志。
        # group_by_length=True,                   #是否根据样本长度对数据进行分组，可以在处理变长序列时提高训练效率。
        #deepspeed='./ds_config_one_gpu.json',    #DeepSpeed 配置文件的路径，用于启用深度学习优化引擎 DeepSpeed。
    )
    
    #step6.初始化 trainer并开始训练
    trainer = Trainer(
        model=model,                              #训练的模型，即 QWenLMHeadModel。
        tokenizer=tokenizer,                      #分词器，用于将文本数据转换为模型输入的 tokens。  
        args=args,                                #训练参数，包括了模型训练的各种设置，如批量大小、学习率、优化器类型等。
        data_collator=data_collator,              #数据收集器，用于准备模型训练所需的输入数据。
        train_dataset=train_dataset,              #训练数据集，包含了用于训练模型的样本数据。
        eval_dataset=eval_dataset,                #评估数据集，包含了用于评估模型性能的样本数据。
        callbacks=[my_trainer_callback],          #回调函数列表，用于在训练过程中插入自定义的操作。
    )

    trainer.train()
    #trainer.train('model_save/pre/checkpoint-3400', resume_from_checkpoint=True)

    #step7.计算困惑度Perplexity& save log
    eval_results = trainer.evaluate()
    print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")

    loss_log = pd.DataFrame(trainer.state.log_history)
    loss_log.to_csv(f"{config.log_dir}/pre_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")

    #step8. 保存模型/lora
    trainer.save_model(args.model_save_dir)
    print('save model or lora adapter to: {}'.format(model_save_dir))


if __name__ == "__main__":

    pretrain_config = PretrainConfig()

    train_pretrain(config=pretrain_config,)






































