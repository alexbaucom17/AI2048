from GeneticAlgorithm import *
import tkinter as tk
import time
import game2048_gui


#MODE = 'Train'
#MODE = 'Train_continue'
MODE = 'Test'

if MODE == 'Train':
    #parameters
    fname = 'CurrentModles/model_20180528_'
    n_agents = 1000
    generations_per_batch = 1
    n_batches = 10
    total_generations = generations_per_batch*n_batches
    print('Preparing to run ' + str(total_generations) +' generations...')

    #instantiate learner
    G = GeneticLearner(n_agents,[16,8,4],seed=4)

    #run batch learning
    for i in range(n_batches):
        print("Starting batch "+str(i))
        G.run_n_generations(generations_per_batch)
        gen_num = i*generations_per_batch
        save_model_state(G,fname + str(gen_num) + '.p')
        print("Model " + str(gen_num) + " Saved")

if MODE == 'Train_continue':
    #define training parameters
    fname = 'CurrentModles/model_20180122_'
    gen_to_load = 1000
    generations_per_batch = 25
    n_batches = 100
    total_generations = generations_per_batch * n_batches

    #load model
    full_filename = fname+str(gen_to_load)+'.p'
    print('Loading ' + full_filename + ' and running ' + str(total_generations) + ' more generations')
    try:
        G = load_model_state(full_filename)
    except:
        raise ValueError('File does not exist: ' + full_filename)

    #restart training
    for i in range(n_batches):
        print("Starting batch "+str(i))
        G.run_n_generations(generations_per_batch)
        gen_num = (1+i)*generations_per_batch + gen_to_load
        save_model_state(G,fname + str(gen_num) + '.p')
        print("Model " + str(gen_num) + " Saved")

if MODE == 'Test':
    #Load model and best agent
    fname = 'CurrentModles/model_20180122_1000'
    n_new_games = 2
    G = load_model_state(fname + '.p')
    A = G.get_best_agent()

    #Initialize gui
    root = tk.Tk()
    root.title("2048 Game")

    #Replay the last game that gave this agent a high score
    score,win = A.replay_previous_game(root)
    print('Agent score: '+str(score)+' Win status: '+ str(win))
    time.sleep(0.5)

    #Play 3 new games to see how the agent performs
    for i in range(n_new_games):
        score, win = A.play_game(root)
        print('Agent score: ' + str(score) + ' Win status: ' + str(win))
        time.sleep(0.5)

    input("Press Enter to end...")