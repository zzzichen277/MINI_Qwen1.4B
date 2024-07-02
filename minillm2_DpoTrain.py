from typing import Dict, Optional
import time
import os 

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import  TrainingArguments,DataCollatorForLanguageModeling
from trl import DPOTrainer,SFTConfig

from peft import LoraConfig, TaskType, PeftModel,get_peft_model
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from qwen.modeling_qwen import QWenLMHeadModel
from qwen.tokenization_qwen import QWenTokenizer

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


from dataclasses import dataclass
from os.path import dirname, abspath


#===================================================================================
# 定义dpo训练配置
@dataclass
class DpoConfig:
    max_seq_len: int = 1024 + 8                  # 8 for eos token
    sft_model_file: str = './model_save/checkpoint-28500_sftmodel_v1_1.4b' # SFT后的模型路径

    tokenizer_dir: str = './model_save/checkpoint-28500_sftmodel_v1_1.4b'   # tokenizer一般和model权重放在同一个文件夹

    dpo_train_file: str = r'./datasets_Processed/dpo_bell_alpaca_train.json' # dpo的训练集
    dpo_eval_file: str = r'./datasets_Processed/dpo_bell_alpaca_eval.json' # dpo的测试集

    #adapter_file: str = './model_save/sft_bellexw/adapter_model.safetensors'
    logs_dpo_dir: str = "./model_save/logs_dpo/"   
    model_save_dir: str = "./model_save/dpo_bell_alpaca/" 

# 处理成dpo数据格式
def get_dataset(split: str, file: str, cache_dir: str = '.cache') -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.
    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }
    """
    dataset = load_dataset('json', data_files=file,  split=split, cache_dir=cache_dir)

    def split_prompt_and_responses(sample: dict) -> Dict[str, str,]:
        return {
            # add an eos token for signal that end of sentence, using in generate.
            "prompt": f"{sample['prompt']}<|im_end|>",
            "chosen": f"{sample['chosen']}<|im_end|>",
            "rejected": f"{sample['rejected']}<|im_end|>",
        }
    return dataset.map(split_prompt_and_responses).shuffle(2333)

class EmptyCudaCacheCallback(TrainerCallback):
    log_cnt = 0
    def on_log(self, args, state, control, logs=None, **kwargs):
        self.log_cnt += 1
        if self.log_cnt % 5 == 0:
            torch.cuda.empty_cache()
                
empty_cuda_cahce = EmptyCudaCacheCallback()

# 定义dpo训练函数
def train_dpo(config: DpoConfig, peft_config: LoraConfig=None) -> None:

    # step 1. 加载训练好的qwen-tokenizer
    tokenizer = QWenTokenizer.from_pretrained(config.tokenizer_dir)

    tokenizer.pad_token_id = tokenizer.im_end_id
    tokenizer.bos_token_id = tokenizer.im_end_id
    tokenizer.eos_token_id = tokenizer.im_end_id

    # step 2. 加载DPO模型-训练好的sft模型checkpoint
    model_train = QWenLMHeadModel.from_pretrained(config.sft_model_file)
    model_ref = QWenLMHeadModel.from_pretrained(config.sft_model_file)

    #step3. 加载DPO训练数据集和评估数据集
    train_dataset = get_dataset("train", file=config.dpo_train_file)
    eval_dataset = get_dataset("train", file=config.dpo_eval_file)
    

    
    """
    configss = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn","c_proj", "w1", "w2" ],
        inference_mode=False,  # 训练模式
        r=8,  # Lora 秩
        lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1,  # Dropout 比例
    )
    
    model = get_peft_model(model, configss)
  


    #step6. 初始化 DPO trainer并开始训练
    
    trainer = Trainer(
        model_train,
        model_ref,
        peft_config=peft_config,
        args=training_args ,
        beta=0.1,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=1032,
        max_target_length=1032,
        max_prompt_length=1032,
        generate_during_eval=True,
        is_encoder_decoder=True,
        # data_collator=data_collator
    )
    
    trainer = Trainer(
        model=model_train,
        tokenizer=tokenizer,
        args=args,
        
        #data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        #callbacks=[empty_cuda_cahce],
        #is_encoder_decoder=True,
    )
    """
 
    
    
    #step5. 初始化DPO训练参数
    args = TrainingArguments(
        output_dir=config.model_save_dir,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        auto_find_batch_size=True,
        remove_unused_columns=False,
        gradient_accumulation_steps=4,
        learning_rate= 1e-5,
        logging_first_step=True, 
        logging_steps= 20,
        save_steps=500,
        optim="adafactor",
        report_to="tensorboard",
        log_level='info',
        warmup_steps=1000,
        bf16=False,
        fp16=True,
        seed=27777,
        logging_dir=config.logs_dpo_dir,
    )
    
    trainer = Trainer(
        model_train,
        args=args,
        #beta=0.1,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        callbacks=[empty_cuda_cahce],
        #max_length=512 * 2 + 16, # 16 for eos bos
        #max_prompt_length=1032,
    )

    
    trainer.train()
    ## resume_from_checkpoint=True

    #step7. save log
    loss_log = pd.DataFrame(dpo_trainer.state.log_history)
    if not os.path.exists(config.logs_dpo_dir):
        os.mkdir(config.logs_dpo_dir)
    loss_log.to_csv(f"{config.logs_dpo_dir}/dpo_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")
    
    #step8. 保存模型/lora
    suffixe = '/lora/' if peft_config is not None else '/dpo'
    model_save_dir = '/'.join(config.sft_model_file.split('/')[0: -1]) + suffixe

    dpo_trainer.save_model(model_save_dir)
    print('save model or lora adapter to: {}'.format(model_save_dir))

   
if __name__ == "__main__":

    dpo_config = DpoConfig()
    train_dpo(dpo_config,peft_config=None )



