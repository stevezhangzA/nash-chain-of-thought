import argparse
import logging
import torch
import random
import time
import os
from utils import *
from sentence_transformers import SentenceTransformer
from local_decoder import custom_api

from collections import Counter

import pickle

import pickle as pkl
from pathlib import Path

def file_to_string(filename):
    # this function is utilized to readout a file for the construction of template.
    # 'Your question is {Question}. Your answer is: {Answer}.'.format(Question='Q:',Answer='A:')
    try:
        with open(filename, 'r') as file:
            return file.read()
    except:
        with open(os.path.join(os.getcwd(), filename), 'r') as file:
            return file.read()
    

class Nash_decoder(object):
    def __init__(self, user_information
                     , game_goalandtips
                     , initial_template
                     , referee_template
                     , answer_filter_template
                     , model
                     , temperature
                     , args
                     , CoT_template=None
                     , selected_players=None
                     , tolerance=1
                     , initial_system=None
                     , direct_answer_trigger=None
                     , LLM=None):
        """
        Multi-Agents Nash Equilibrium System
        """
        # initialize contextual information
        # self.pre_request= pickle.load(open(player_instruction,'rb'))          # instructions: construct agent.
        self.optimal_player_id=0
        self.user_information = pickle.load(open(user_information, 'rb'))  # template for building the users.
        self.pre_request = file_to_string(game_goalandtips)  # the goal and rules in this game.
        self.players_template = file_to_string(initial_template)  # initialize the template for each players.
        if initial_system != None:  # templates used to construct the game rules.
            self.initial_system = file_to_string(initial_system)
        # referee_template
        self.referee_template = file_to_string(referee_template)
        self.answer_filter_template = file_to_string(answer_filter_template)
        # initialize the parameters of nash system:
        self.tolerance = tolerance  # the upperbound of iterations when seeking for a reach of nash equilibrium.
        # cosntruct the players' contextual information
        if CoT_template != None:  # chain of thought prompt
            cot_prompt = file_to_string(CoT_template)
        else:
            cot_prompt = "Let's think step by step."
        self.cot_prompt = cot_prompt
        if direct_answer_trigger != None:
            self.direct_answer_trigger = direct_answer_trigger  # prompts to obtain the last answer
        else:
            self.direct_answer_trigger = 'Therefore the answer is: '
        # initializing players
        self.player_pool = {}  # initialize the context information for all players
        selected_players = list(self.user_information.keys())[:selected_players]
        for player_key in selected_players:
            print(player_key, ': ', self.user_information[player_key])
            if selected_players != None:
                if player_key not in selected_players:
                    continue
            c_template = self.players_template
            self.player_pool[player_key] = c_template
        # all initialized players
        tag_ = ['0)', '1)', '2)', '3)', '4)', '5)', '6']
        new_selected_players = []
        for id_ in range(len(selected_players)):
            new_selected_players.append(tag_[id_] + selected_players[id_])
        if selected_players != None:
            all_players = ', '.join(new_selected_players)
        else:
            all_players = ', '.join(list(self.player_instruction.keys()))
        # players and gaming rules
        self.all_players = all_players
        self.all_player_list = selected_players
        self.initial_system.format(all_players=all_players)
        # initializing the conversation tag
        self.question_tag = 'Q: '
        self.answer_tag = 'A: '
        # initializing the API parameters
        self.model = model
        self.temperature = temperature
        # initializing sentence model
        self.encoder = SentenceTransformer(args.encoder)
        self.args = args
        # initialize sentence model
        self.args = args
        self.LLM = LLM

    def point_the_optimal_agent(self, query, max_length=None):
        messages = [{"role": "user",
                     "content": f'current issue is {query}, and the best player is who? (please give us the number of that player):'}]
        response_cur = self.LLM.inference(messages)
        return re.findall('[0-9]', response_cur)

    def point_the_optimal_agent_with_cot(self, query, max_length=None):
        choices = [str(id_) + '.' for id_ in range(100)]
        constructed_options = []
        for player_id in range(len(self.all_player_list)):
            constructed_options.append(choices[player_id] + self.all_player_list[player_id])
        constructed_options = ' '.join(constructed_options)
        refree_question = f'Q: current issue is {query}, and the best player is who? Please give us the number of that player from the options below: {constructed_options}'
        initial_system = {"role": "system",
                          "content": f'There are total {len(self.players_template)} players including {self.all_players}. Please point out the most appropriate player for the following task:i\n'}
        messages = [{"role": "user",
                     "content": initial_system["content"] + refree_question + '\n' + 'A: Let us think step by step.'}]
        z = self.LLM.inference(messages)
        messages = [{"role": "user",
                     "content": initial_system["content"] + refree_question + '\n' + 'A: Let us think step by step.' + z + 'Therefore, the most appropriate player in this game is who? (please direct give us the number)'}]
        response_cur = self.LLM.inference(messages)
        return re.findall('[0-9]', response_cur)

    def point_the_optimal_answer_with_cot(self,query,answer_list):
        options=['0)','1)','2)','3)','4)','5)']
        new_candidates=[options[i]+answer_list[i] for i in range(len(answer_list))]
        bound_op=options[len(answer_list)-1]
        answer_filter=self.answer_filter_template.format(player=self.all_player_list[self.optimal_player_id],
                                                         instruction=self.user_information[self.all_player_list[self.optimal_player_id]],
                                                         query=query,
                                                         options=', '.join(new_candidates))
        messages = [{"role": "user",
                     "content": answer_filter}]
        z = self.LLM.inference(messages) 
        while(1):
            messages = [{"role": "user",
                     "content": answer_filter + z + f'Therefore, among 0) throught {bound_op}, the answer is'}]
            response_cur = self.LLM.inference(messages)
            optimal_choice=re.findall('[0-9]',response_cur)
            try:
                id_=re.findall('[0-9]', response_cur)
                pred=answer_list[int(id_[-1])]
                break
            except:
                pass
        return pred,z

    def answer_filter(self,query,answer_list,args):

        message=[{"role": "user",
                 "content": self.initial_system + '\n' + self.player_pool[self.all_player_list[self.optimal_player_id]].format(query=self.question_tag + query,
                                                                                                                               player_key=self.user_information[list(self.player_pool.keys())[self.optimal_player_id]],
                                                                                                                               cot_prompt='A: ' + self.cot_prompt)}]
        z= self.LLM.inference(message)
        message=[{"role": "user",
                 "content": self.initial_system + '\n' + self.player_pool[self.all_player_list[self.optimal_player_id]].format(query=self.question_tag + query,
                                                                                                                               player_key=self.user_information[list(self.player_pool.keys())[self.optimal_player_id]],
                                                                                                                            cot_prompt='A: ' + self.cot_prompt+ z + ' ' + self.direct_answer_trigger)}]
        answer = self.LLM.inference(message)
        answer=answer_cleansing(args, answer)
        return [answer_list, answer]
    

    #def reach_nash(self, query, args, answers, max_length):
    #    print("point out the most appropriate player")
    #    while 1:
    #        try:
    #            optimal_player_id = self.point_the_optimal_agent_with_cot(query, max_length=max_length)
    #            self.optimal_player_id = int(eval(optimal_player_id[-1]))
    #            if self.optimal_player_id not in list(range(len(self.all_player_list))):
    #                continue
    #            break
    #        except:
    #            pass
    #    return self.answer_filter(answer_list=answers,query=query,args=args)
    
    def confine_player(self,query,max_length):
        print("point out the most appropriate player")
        while 1:
            try:
                optimal_player_id = self.point_the_optimal_agent_with_cot(query, max_length=max_length)
                self.optimal_player_id = int(eval(optimal_player_id[-1]))
                if self.optimal_player_id not in list(range(len(self.all_player_list))):
                    continue
                break
            except:
                pass

    def answer_clean(self, pre_answer , args):
        pred = answer_cleansing(args, pre_answer)
        return pred

    def answer_clean(self, pre_answer , args):
        pred = answer_cleansing(args, pre_answer)
        return pred

def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    fix_seed(args.random_seed)
    if args.CoT_template == 'None':
        cot_template = None
    else:
        cot_template = args.CoT_template
    decoder = custom_api(
        tokenizer_path=args.tokenizer_path
        , pretrained_model=args.model_tag
    )
    # decoder=[]
    decoder = Nash_decoder(args.user_information, args.game_goalandtips, args.initial_template,
                           args.referee_template, args.answer_filter_template,args.engine_model, args.temperature, args, CoT_template=cot_template,
                           selected_players=args.selected_players, tolerance=args.tolerance,
                           initial_system=args.initial_system,
                           direct_answer_trigger=args.direct_answer_trigger_for_zeroshot_cot,
                           LLM=decoder)
    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()
    if args.method == "few_shot":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method == "few_shot_cot":
        demo = create_demo_text(args, cot_flag=True)
    else:
        pass
    total = 0
    correct_list = []
    record = []
    for i, data in enumerate(dataloader):
        print('*************************')
        print("{}st data".format(i + 1))
        # Prepare question template ...
        x, y = data
        x = "Q: " + x[0] + "\n" + "A:"
        y = y[0].strip()
        max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
        all=[]
        decoder.confine_player(x,max_length)
        for local_it in range(3):
            answers=[]
            for iterations in range(2):
                # Prepare question template ...
                x, y = data
                x = "Q: " + x[0] + "\n" + "A:"
                y = y[0].strip()
                if args.method == "zero_shot":
                    x = x + " " + args.direct_answer_trigger_for_zeroshot
                elif args.method == "zero_shot_cot":
                    x = x + " " + args.cot_trigger
                elif args.method == "few_shot":
                    x = demo + x
                elif args.method == "few_shot_cot":
                    x = demo + x
                else:
                    raise ValueError("method is not properly defined ...")
                # Answer prediction by generating text ...
                max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
                z = decoder.LLM.inference( [{"role": "user", "content": x }])
                # Answer extraction for zero-shot-cot ...
                if args.method == "zero_shot_cot":
                    z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
                    max_length = args.max_length_direct
                    pred = decoder.LLM.inference([{"role": "user", "content": z2}])
                    #record.append(z2 + pred)
                    print(z2 + pred)
                else:
                    pred = z
                    #record.append(x+z)
                    print(x + pred)
                pred = answer_cleansing(args, pred)
                answers.append(pred)
            filtered_answers = decoder.reach_nash(x, args,answers, max_length=max_length)
            all.append(filtered_answers)
        early_stop=False
        record={}
        all_pre=[]
        for ans in all:
            answer_list, answer_star=ans[0],ans[1]
            if answer_star in answer_list:
                if answer_star not in record:
                    record[answer_star]=0
                else:
                    record[answer_star]+=1

            all_pre.extend([answer_star,answer_list[0],answer_list[1]])

        if record!={}:
            pred = max(record, key=lambda x: record[x])
        else:
            counter = Counter(all_pre)
            most_common_element = counter.most_common(1)[0][0]
            # print(god_rational)
            pred = most_common_element

        print("pred : {}".format(pred))
        print("GT : " + y)
        print('*************************')
        # Checking answer ...
        correct = (np.array([pred]) == np.array([y])).sum().item()
        correct_list.append(correct)
        total += 1  # np.array([y]).size(0)
        # Calculate accuracy ...
        accuracy = (sum(correct_list) * 1.0 / total) * 100
        print("accuracy : {}".format(accuracy))
        if not os.path.exists(os.path.join('loggings',args.proj_name,args.sub_dir,args.method,args.dataset,str(args.random_seed))):
            nested_folder = Path(os.path.join('loggings',args.proj_name,args.sub_dir,args.method,args.dataset,str(args.random_seed)))
            nested_folder.mkdir(parents=True)
        with open(os.path.join('loggings',args.proj_name,args.sub_dir,args.method,args.dataset,str(args.random_seed),'training_results.pkl'),'wb') as f:
            pkl.dump({'acc':accuracy,'data':record},f)
        if i>=args.capacity_one_epoch:
            break

def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")
    parser.add_argument(
        "--api_log_file_name", type=str, default=None,
        help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]"
    )
    parser.add_argument('--tokenizer_path', type=str, default='person_tokenizer.pkl')
    parser.add_argument('--model_tag', type=str,
                        default='/zhangziqi/self_llm/hf_hub/models--mistralai--Mistral-7B-Instruct-v0.2')
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument("--proj_name", type=str, default="nash_cot")
    parser.add_argument("--sub_dir",type=str,default="Mistral_7B")
    parser.add_argument(
        "--dataset", type=str, default="aqua",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq",
                 "bigbench_date", "object_tracking", "coin_flip", "last_letters"], help="dataset used for experiment"
    )
    parser.add_argument(
        "--capacity_one_epoch",type=int, default=60)

    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1],
                        help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")

    parser.add_argument("--max_num_worker", type=int, default=3, help="maximum number of workers for dataloader")

    parser.add_argument(
        "--model", type=str, default="gpt3", choices=["gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl"],
        help="model used for decoding. Note that 'gpt3' are the smallest models.")
        
    parser.add_argument(
        "--method", type=str, default="zero_shot_cot",
        choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot"], help="method"
    )
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1,
        help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=128,
        help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32,
        help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=10,
        help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    parser.add_argument(
        '--game_rules', type=str, default='prompts/game_rules.txt', help='game rules and goal for all players'
    )
    parser.add_argument(
        '--initial_system', type=str, default="prompts/initial_system.txt", help='template of each player'
    )
    parser.add_argument(
        '--initial_user', type=str, default="prompts/initial_user.txt", help='template of each player'
    )
    parser.add_argument(
        '--player_instruction', type=str, default='prompts/player_instruction.json',
        help='this is the introduction of list of multi players'
    )
    parser.add_argument(
        '--user_information', type=str, default='prompts/player_instruction.pkl',
        help='this is a list of multi players introduction'
    )
    parser.add_argument(
        '--game_goalandtips', type=str, default='prompts/game_goalandtips.txt', help='Rules of game'
    )
    parser.add_argument(
        '--initial_template', type=str, default='prompts/initial_template.txt', help='Blank template for any agents'
    )
    parser.add_argument(
        '--referee_template', type=str, default='prompts/referee_template.txt', help='Prompt for the referee Agent'
    )
    parser.add_argument(
        '--CoT_template', type=str, default='prompts/cot_template.txt', help='Chain of thought prompt'
    )
    parser.add_argument(
        '--selected_players', type=int, default=4, help='this is a list of multi players introduction'
    )
    parser.add_argument(
        '--tolerance', type=int, default=3, help='this is a list of multi players introduction'
    )
    parser.add_argument(
        '--engine_model', type=str, default='gpt-3.5', help='this is a list of multi players introduction',
        choices=["gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl", "code-davinci-002", 'gpt-3.5', 'gpt-3.5-turbo',
                 'text-davinci-002']
    )
    parser.add_argument(
        '--encoder', type=str, default='sentence_transformer_/models--sentence-transformers--all-MiniLM-L6-v2',
        help='used to extract sentence embedding'
    )
    parser.add_argument(
        '--openai_key', type=str, default="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"
    )
    parser.add_argument(
        '--n_questions', type=int, default=10
    )
    parser.add_argument(
        '--temperature', type=int, default=0
    )
    parser.add_argument(
        '--answer_filter_template', type=str, default='prompts/answer_filter_template.txt'
    )
    args = parser.parse_args()
    if args.dataset == "aqua":
        args.dataset_path = "dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    if args.cot_trigger_no == 1:
        args.cot_trigger = "Let's think step by step."
    elif args.cot_trigger_no == 2:
        args.cot_trigger = "We should think about this step by step."
    elif args.cot_trigger_no == 3:
        args.cot_trigger = "First,"
    elif args.cot_trigger_no == 4:
        args.cot_trigger = "Before we dive into the answer,"
    elif args.cot_trigger_no == 5:
        args.cot_trigger = "Proof followed by the answer."
    elif args.cot_trigger_no == 6:
        args.cot_trigger = "Let's think step by step in a realistic way."
    elif args.cot_trigger_no == 7:
        args.cot_trigger = "Let's think step by step using common sense and knowledge."
    elif args.cot_trigger_no == 8:
        args.cot_trigger = "Let's think like a detective step by step."
    elif args.cot_trigger_no == 9:
        args.cot_trigger = "Let's think about this logically."
    elif args.cot_trigger_no == 10:
        args.cot_trigger = "Let's think step by step. First,"
    elif args.cot_trigger_no == 11:
        args.cot_trigger = "Let's think"
    elif args.cot_trigger_no == 12:
        args.cot_trigger = "Let's solve this problem by splitting it into steps."
    elif args.cot_trigger_no == 13:
        args.cot_trigger = "The answer is after the proof."
    elif args.cot_trigger_no == 14:
        args.cot_trigger = "Let's be realistic and think step by step."
    else:
        raise ValueError("cot_trigger_no is not properly defined ...")
    return args
if __name__ == "__main__":
    main()



