from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
try:
    from modelscope import Model, snapshot_download
    from modelscope.models.nlp.llama2 import Llama2Tokenizer
except:
    pass
import torch

class custom_api(object):
    def __init__(self,tokenizer_path=None
                     ,pretrained_model=None
                     ,device="cuda"):
        self.device = device
        try:
            self.tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,
                                                         trust_remote_code=True)
        except:
            self.tokenizer = pickle.load(open(tokenizer_path,'rb'))
        
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model)
        self.model.to(self.device)

    def inference(self,messages,max_new_tokens=256):
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=max_new_tokens, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        #return decoded[0].split('[/INST]')[-1].strip('</s>')
        return decoded[0]

    def inference_raw(self,messages):
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)
        return decoded

class custom_api_others(object):
    def __init__(self,tokenizer_path=None
                     ,pretrained_model=None
                     ,device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model,
                                                       trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model,
                                                          trust_remote_code=True)
        self.model.to(self.device)

    def inference(self,messages,max_new_tokens=256:
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=max_new_tokens, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded[0]

    def inference_raw(self,messages):
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)
        return decoded


    def inference_parallel(self,messages):
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=256, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)
        return [decoded[i_].strip('<|end_of_turn|>') for i_ in range(len(decoded))]



