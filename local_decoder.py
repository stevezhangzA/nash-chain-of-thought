from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
try:
    from modelscope import Model, snapshot_download
    from modelscope.models.nlp.llama2 import Llama2Tokenizer
except:
    pass
import torch


