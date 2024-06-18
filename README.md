# Nash Chain-of-Thought (CoT)

*Description:* This is the official codebase of Nash CoT

### What's CoT

CoT is a *step-by-step* manner inference approach. 

Here, we provide a case of CoT's template: 
- *question: x*,  
- *prompt: 'Let's think step by step'*, 
- *trigger: 'Therefore, the answer is:'* 

Meanwhile, this approach is composed of two steps:

- Step1 (obtain rational): *z<-LLM(z|x, prompt)* 
- Step2 (obtain answer)  : *a<-LLM(a|x, prompt, z, trigger)*

### Multi-path inference with CoT

Previously, self-consistency[2] demonstrates that using multi-path CoT inference with voting for the highest frequency answer can improve the accuracy of predictions. However, its computation is insufficient to conduct multi-path inference. Therefore, we propose Nash CoT to solve this limitation.

### The framework of Nash CoT:

Nash CoT uses question-related contextual information as a template for making inferences in each path. It also utilizes Preference Equilibrium to reduce overconfident generation. Specifically, as illustrated below, if the generation is guided by the template during mini-batch inference and has been labeled, it indicates that the generation has reached preference equilibrium. Ultimately, we identify the highest frequency that has reached preference equilibrium.

![image](demonstration.png)

## Configuration and experiment
### Configuration
```c
# python version          : 3.8
# system                  : Linux
# PyTorch                 : 2.0.0
# Huggingface Transformer : 4.38.2
conda create --name nash_cot python=3.8
conda activate nash_cot
unzip nash-chain-of-thought.zip
cd nash-chain-of-thought && pip install -r requirements.txt
```
### how to download LLM
```c
# Through these commands you can download LLM from huggingface transformer
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",cach_dir='your path')
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",cach_dir='your path')
# However, some LLMs require specific command, such that GLM4 has to set trust_remote_code=True,
```
### Experiment
```c
# sh run_nash_cot.sh dataset random_seed tokenizer_path model_path
#  in particular,
#  $1: datasets, all datasets' name are listed in the configuration in nash_cot.py
#  $2: random_seed (int)
#  $3: model_path, we utilize huggingface to localy deploy LLM, model_path represent catche_dir 
#  $3: tokenizer_path: same as model_path
#  $4: outer loop
#  $5: inner loop
# subsquently, we provide a case below:
sh run_nash_cot.sh aqua 0 ./hf_hub/models--mistralai--Mistral-7B-Instruct-v0.2 3 2
sh run_nash_cot.sh aqua 2 ./hf_hub/models--mistralai--Mistral-7B-Instruct-v0.2 3 2
sh run_nash_cot.sh aqua 4 ./hf_hub/models--mistralai--Mistral-7B-Instruct-v0.2 3 2
```

### We list several uncontrollable factors during the evaluation process:

This means that if you can address these uncontrollable factors (designing a better template etc.), your experimental performance may surpass ours.

- We found that each LLM has its own unique characteristics. For instance, given a prompt, Llama3 can directly answer the question, whereas Mistral-Instruct (7B) requires a predefined trigger to guide it to the final answer. Therefore, you can design specific templates tailored for Llama3.
- We found that player templates significantly impact the performance of LLMs. Therefore, we encourage users to explore more effective templates to further enhance the performance of Nash CoT.

### Introduction of our player template

we totally set up 6 roles:

role      | player template 
--------  | --------------------- 
mathematician | You are a mathematician, you excel at analyzing problems from a mathematical logical perspective and arrive at conclusions that align with your values.
litterateur   | You are a literary scholar who has read a vast array of literary works. Please consider the problem from the perspective of a literary scholar.
Philosopher   | You are a philosopher, your knowledge base includes a wealth of philosophical knowledge. You enjoy approaching problems from a philosophical perspective and arriving at conclusions that align with your values.
geographer    | You are a geographer with a deep understanding of geographical knowledge. Please approach the given problem from the perspective of a geographer.
statesman     | You are a politician, and your decision-making process stems from your role as a politician. Please make decisions based on this perspective regarding the given problem.

### Meanwhile, we provide some our runned loggings

Model     | Method | AQuA | GSM8K| Coin Flip |Object Tracking | Bigbench Date|CommonsensQA|
--------  | ----- | ----- | ----- |----- |----- |-----|-----|
Mistralai-Instruct (7B) | self-consistency |34.4 $\pm$ 6.1 | 58.5 $\pm$ 2.8|  21.9 $\pm$ 4.7|38.8 $\pm$ 0.8|47.0 $\pm$ 1.5 | 71.0 $\pm$ 3.4|
Mistralai-Instruct (7B) | Nash CoT |39.9 $\pm$ 5.4| 55.7 $\pm$ 5.8 |29.0 $\pm$ 5.4 | 44.8 $\pm$ 2.0 | 41.1 $\pm$ 1.2| 69.4 $\pm$ 4.7|

We selected several representative tasks from Arabic reasoning, symbolic inference, and CommonsenseQA, and saved them in a folder named "logging".

## *Thanks* 

We use the following models: GLM4-chat (9B)[4], Mistral-Instruct (7B)[5]. We have also introduced Nash CoT with LLama3-Instruct [6]. However, there are concerns regarding uncontrollable factors mentioned above. 

Our codebase has been derived from Automatic CoT's codebase, and our local decoder has been constructed using Huggingface Transformer.

Addtionally, thanks my collaberator: Cunxiang Wang

This reseach is sponsed by MiLab at WestLake Univeristy.

*If you utilize our codebase, please cite below:*
```c
@article{,
    title={Nash CoT: Multi-Path Inference with Preference Equilibrium}, 
    author={Ziqi Zhang and Cunxiang Wang and Xiong Xiao and Yue Zhang and Donglin Wang},
    year={2024},
    eprint={},
    archivePrefix={arXiv}
}
```
# Collaberation 

If you discover any new uses for Nash CoT, please feel free to contact us and provide new experimental results. For example: 1) You conduct an evaluation on LLama, GPT, etc. 2) You find that this approach can be scaled to another setting, such as controllable generation.

Both my email (stevezhangz98a@gmail.com) and adding a new blog to this project are welcome.

## *Reference*

[1] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou. 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. Preprint. arXiv: 2201.11903.

[2] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou. 2022. Self-Consistency Improves Chain of Thought Reasoning in Language Models. Preprint. arXiv: 2203.11171.

[3] Zhuosheng Zhang, Aston Zhang, Mu Li, Alex Smola. 2022. Automatic Chain of Thought Prompting in Large Language Models. Preprint. arXiv: 2210.03493.

[4] Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Peng Zhang, Yuxiao Dong, Jie Tang. 2022. GLM-130B: An Open Bilingual Pre-trained Model. Preprint. arXiv: 2210.02414.

[5] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed. 2022. Mistral 7B. Preprint. arXiv: 2310.06825.

