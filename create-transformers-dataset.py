# coding: utf-8
#
# Filename: create-transformers-dataset.py
# Author: Reto Gubelmann
# Date modified: 2020/07/29
# Input files needed: None, input: None
#
# (c) 2020 Reto Gubelmann
#
#************************************************************
# Functionality
#
# The Script 
# 1. takes a line-by-line textfile 
# 2. creates a transformers-compatible dataset
# 3. Saves it to disk
#
# Use "finetune-transformer-model.py" to finetune based on dataset
#
#*************************************************************
# Sample call
#
# (remember to activate virtualenv)
#
# nohup python3 create-transformers-dataset.py &
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
from transformers import DataCollatorForLanguageModeling, RobertaTokenizerFast
from transformers import BertTokenizer, BertTokenizerFast, BertModel, BertForMaskedLM, AutoModel, AutoTokenizer, AutoModelWithLMHead
from transformers import LineByLineTextDataset
import torch
from torch.utils.data import DataLoader


# Check that PyTorch sees Cuda

print("Is it true that Cuda is available?",torch.cuda.is_available())


# Specify tokenizer

#model = AutoModelWithLMHead.from_pretrained('t5-large') # TOGGLE FOR EN-GER
#tokenizer = AutoTokenizer.from_pretrained('bert-large')
#tokenizer = RobertaTokenizerFast.from_pretrained('../data/Models/vocab-bert-large-de', do_lower_case=False)
tokenizer = AutoTokenizer.from_pretrained('bert-base-german-dbmdz-cased')
#tokenizer=tokenizer

# Build dataset from Textfile given

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
#    file_path="../data/Input/input_100_Neg_en_JCKREG.txt",
    file_path="./Input/input_466k_de_wiki20200620.txt",
    block_size=128,
)

# Save dataset

torch.save(dataset, './Input/466kwiki.pt')