import torch
import pickle
import os

def load_data():
    """You can use this function to load the files you saved using 
    Atari_DeepMind_Grid"""
    game_name = input("""Enter the name of the game from which you want to load 
                          data (without version) : """)
    model = input("""model (dqn/resnet) : """)
        
    if os.path.exists("evaluation_model_{}_{}.txt".format(game_name,model)):
        with open("evaluation_model_{}_{}.txt".format(game_name,model), 'rb') as fichier:
            mon_depickler = pickle.Unpickler(fichier)
            evaluation_model = mon_depickler.load()
            fichier.close()
    else:
        evaluation_model = None
    
    if os.path.exists("current_results_{}_{}.txt".format(game_name,model)):
        with open("current_results_{}_{}.txt".format(game_name,model), 'rb') as fichier:
            mon_depickler = pickle.Unpickler(fichier)
            current_results = mon_depickler.load()
            fichier.close()
    else:
        current_results = None
        
    if os.path.exists("run_time_{}_{}.txt".format(game_name,model)):
        with open("run_time_{}_{}.txt".format(game_name,model), 'rb') as fichier:
            mon_depickler = pickle.Unpickler(fichier)
            time = mon_depickler.load()
            fichier.close() 
    else:
        time = None
        
    if os.path.exists("model_atari_{}_{}.pkl".format(game_name,model)):
        model = torch.load("model_atari_{}_{}.pkl".format(game_name,model))
    else:
        model=None
    
    return evaluation_model, current_results, time, model



if __name__ == "__main__":
    while True:
        a,b,c,d = load_data()
        print(a,b,c,d)
        continuer = input("Want to load more data ? (y/n) ").lower()
        if continuer!='y':
            break
        
    
    


