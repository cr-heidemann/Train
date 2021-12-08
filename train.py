from tokenizers import BertWordPieceTokenizer, CharBPETokenizer, ByteLevelBPETokenizer, SentencePieceBPETokenizer
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast, BertConfig, BertForPreTraining
from tokenizers.implementations import ByteLevelBPETokenizer, CharBPETokenizer, SentencePieceBPETokenizer, BertWordPieceTokenizer
from tokenizers.processors import BertProcessing
import numpy as np
from datasets import load_metric, load_dataset

from transformers import TrainingArguments, Trainer
#import wandb

"""wandb.init(project="my-test-project", entity="heidemann")

wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128
}"""


#Load pre-trained Tokenizer
print("Load Tokenizer")
"""mytokenizer=PreTrainedTokenizerFast(tokenizer_file="./Tokenizer/sent-BPE.tokenizer.json",
                                    mask_token="<mask>",
                                    #pad_token="<pad>",
                                    unk_token="<unk>",
                                    cls_token="<s>",
                                    sep_token="</s>",
                                    max_length=500,
                                    truncation=True,
                                    padding=True,
                                    return_tensors="pt"

                                    )
mytokenizer.add_special_tokens([ "<pad>", "<mask>" ])"""
tokenizer=PreTrainedTokenizerFast(tokenizer_file="./Tokenizer/sent-BPE.tokenizer.json",    #AutoTokenizer.from_pretrained(pretrained_model_name_or_path="./Tokenizer/bertword_tokenize"
                                    mask_token="<mask>",
                                    pad_token="<pad>",
                                    unk_token="<unk>",
                                    bos_token="<s>",
                                    eos_token="</s>"
                                    )                                    
print(tokenizer)
#Initialise model
print("Initialise model")


config_path="./Config/BertBase/config.json"
config=BertConfig.from_pretrained(config_path)
model=BertForPreTraining(config)
"""from transformers import AutoTokenizer, AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")"""
#print('No of parameters: ', model.num_parameters())
#print(model.parameters)

#Dataset
data="./Data/Alles510.txt"
print("Make Dataset")
from transformers import TextDatasetForNextSentencePrediction
dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer,
    file_path=data,
    block_size = 256
)



from transformers import DataCollatorForLanguageModeling
print("Data Collator")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True,
    mlm_probability= 0.15
)
print("Train Model")
from transformers import Trainer, TrainingArguments

#wandb.log({"loss": loss})

# Optional
#wandb.watch(model)
training_args = TrainingArguments(
    output_dir= "./Working",
    overwrite_output_dir=True,
    save_steps=10_000,
    save_total_limit=2,
    per_device_train_batch_size=16,
    num_train_epochs=2,
    #per_gpu_train_batch_size= 16,
    prediction_loss_only=True,
    #report_to="wandb"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    #eval_dataset=dataset,
    #compute_metrics=compute_metrics
)

trainer.train()

#Save Model
print("Save")
trainer.save_model("./Modelle/BertBase/sent_base_2.model")
#trainer.evaluate()
print("Done")

