import pickle as pkl

data=pkl.load(open('player_instruction.pkl','rb'))
for k in data:
    print(data[k])
