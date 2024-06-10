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
        self.tokenizer = pickle.load(open(tokenizer_path,'rb'))
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model)
        self.model.to(self.device)

    def inference(self,messages):

        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(self.device)

        generated_ids = self.model.generate(model_inputs, max_new_tokens=256, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)

        return decoded[0].split('[/INST]')[-1].strip('</s>')

    def inference_raw(self,messages):

        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(self.device)

        generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)

        return decoded


    def inference_parallel(self,messages):

            #response_cur = self.LLM.inference(c_message)
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(self.device)

        generated_ids = self.model.generate(model_inputs, max_new_tokens=256, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)

        return [decoded[i_].split('[/INST]')[-1].strip('</s>') for i_ in range(len(decoded))]



class custom_api(object):
    def __init__(self,tokenizer_path=None
                     ,pretrained_model=None
                     ,device="cuda"):
        self.device = device
        self.tokenizer = pickle.load(open(tokenizer_path,'rb'))
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model)
        #tokenizer = AutoTokenizer.from_pretrained("D:\\chain_of_thought\\awesome_chain_of_thought\\HuggingFace-Download-Accelerator\\hf_hb\\models--mistralai--Mistral-7B-Instruct-v0.2")
        self.model.to(self.device)

    def inference(self,messages):

        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(self.device)

        generated_ids = self.model.generate(model_inputs, max_new_tokens=256, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)

        return decoded[0].split('[/INST]')[-1].strip('</s>')

    def inference_raw(self,messages):

        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(self.device)

        generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)

        return decoded


    def inference_parallel(self,messages):

            #response_cur = self.LLM.inference(c_message)
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(self.device)

        generated_ids = self.model.generate(model_inputs, max_new_tokens=256, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)

        return [decoded[i_].split('[/INST]')[-1].strip('</s>') for i_ in range(len(decoded))]


class custom_api_ms(object):
    def __init__(self,tokenizer_path=None
                     ,pretrained_model=None
                     ,device="cuda"):

        model_dir = snapshot_download("modelscope/Llama-2-13b-chat-ms", revision='v1.0.2',
                              ignore_file_pattern=[r'.+\.bin$'])
        #print(model_dir)
        self.tokenizer = Llama2Tokenizer.from_pretrained(model_dir)
        self.model = Model.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map='auto')


    def inference(self,messages):
        if isinstance(messages,list):
            messages=messages[0]['content']
        inputs = {'text': messages,
                'system':'user',
                'max_length': 2048}

        output = self.model.chat(inputs, self.tokenizer)
        #print(output)
        #print(type(output))
        return output['response']

