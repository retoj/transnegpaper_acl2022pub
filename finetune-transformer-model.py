# coding: utf-8
#
# Filename: finetune-transformer-model.py
# Author: Reto Gubelmann
# Date modified: 2020/07/29
# Input files needed: None, input: None
#
# (c) 2020 Reto Gubelmann
#
#************************************************************
# Functionality
#
# The Script is a substantially adapted version of the 
# excellent colab-Notebook located here:
# https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb#scrollTo=IMnymRDLe0hi
# It does the following:
# 1. Read in a dataset
# 2. Finetune given transformer model
# 3. Save the Model to Disk.
#
# Use "create-transformers-dataset.py" to create dataset
#
#*************************************************************
# Sample call
#
# (remember to activate virtualenv)
#
# nohup python3 finetune-transformer-model.py &
#
#*************************************************************
# Credits
# As mentioned: The Guys who wrote the Colab-Notebook
# Aditya Malte and Probably somebody else?
# 
#Begin of Program


# Import Modules, etc.

import os
import sys


from transformers import DataCollatorForLanguageModeling
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AutoModel, AutoTokenizer, AutoModelWithLMHead
from transformers import LineByLineTextDataset, get_linear_schedule_with_warmup
from transformers import Trainer, TrainingArguments
import apex
from torch.utils.data import DataLoader
import torch
from apex.optimizers import FusedLAMB
from torch.optim.lr_scheduler import LambdaLR

# Check Check Cuda-Availability

print("Is it true that Cuda is available?",torch.cuda.is_available())


# Specify model and tokenizer

model = AutoModelWithLMHead.from_pretrained('bert-base-german-dbmdz-cased') # TOGGLE FOR EN-GER
tokenizer = AutoTokenizer.from_pretrained('bert-base-german-dbmdz-cased')

# Put Model to Cuda

model=model.cuda()

# Load Dataset

dataset = torch.load("./Input/466kwiki.pt")

# Specify Data Collator


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False #, mlm_probability=0.15
)

# Optimnizer & Scheduler

opt = FusedLAMB(model.parameters(), lr = 6e-3) # For testing deactivated

#sched=LambdaLR(opt, lr_lambda=lambda1)
#sched = LambdaLR(opt, lr_lambda=[lambda epoch: 0.95 ** epoch])
#sched=get_linear_schedule_with_warmup(opt, num_warmup_steps=280, num_training_steps=780) # For testing adapted. Originally: 280, 98000
sched=get_linear_schedule_with_warmup(opt, num_warmup_steps=100, num_training_steps=10000) # For testing adapted. Originally: 280, 98000
# ### Finally, we are all set to initialize our Trainer

# Specify Trainer

training_args = TrainingArguments(
    output_dir="../data/Models/20200925-Neg-bert-dbmdz-466kwiki-exttrain",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    fp16=False,
    
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
    optimizers= (opt,sched), # for testing deactivated
)





# This part: Use amp or not.

## Currently not using it, tests show that transformer-trainer is
## Already very fast, no gains obtained.

#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
#model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O2")
#model = torch.nn.DataParallel(model) # New, perhaps crashes...

# Train

trainer.train()

# Save final model (+ tokenizer + config) to disk

trainer.save_model("../data/Models/20200925-Neg-bert-dbmdz-466kwiki-exttrain")

