import neat
import time
import pickle
import sys
import glob
import os
import cv2
import numpy as np
import threading
from PIL import ImageGrab
from selenium import webdriver
from pynput.keyboard import Key, Controller, Listener

home = os.path.dirname(os.path.abspath(__file__))
match_threshold = 0.8

generation = 0
max_fitness = 0
best_genome = 0
score = 0

moves = 0

dino_traces = []
files = glob.glob(home + '\Object_screenshots\Dino*.jpg')
for file in files:
    temp = cv2.imread(file, 0)
    dino_traces.append(temp)

object_traces = []
files = glob.glob(home + '\Object_screenshots\Cactus*.jpg')
for file in files:
    temp = cv2.imread(file, 0)
    object_traces.append(temp)

files = glob.glob(home + '\Object_screenshots\Ptero*.jpg')
for file in files:
    temp = cv2.imread(file, 0)
    object_traces.append(temp)

def on_press(key):
    pass

def on_release(key):
    global moves
    print 'ACTION!!!!!'
    moves += 1
    return False

def game(genome, config):

    global moves

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    driver = webdriver.Chrome()
    driver.set_window_size(240, 320)
    driver.get("chrome://dino/")
    keyboard = Controller()
    keyboard.press(Key.space)

    time.sleep(2)

    global score
    global object_coord
    object_coord = 0
    global dino_coord
    dino_coord = 0

    object_coord = (0, 0)
    start_time = time.time()

    while True:

        listener = Listener(on_press=on_press, on_release=on_release)
        listener.start()

        screen = np.array(ImageGrab.grab(bbox=(25, 135, 490, 320)))
        screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        for dino in dino_traces:
            res_dino = cv2.matchTemplate(screen_gray, dino, cv2.TM_CCOEFF_NORMED)
            loc_dino = np.where(res_dino >= match_threshold)
            for dino_tr in zip(*loc_dino[::-1]):
                dino_coord = dino_tr[0], dino_tr[1]

        for object in object_traces:
            res_object = cv2.matchTemplate(screen_gray, object, cv2.TM_CCOEFF_NORMED)
            loc_object = np.where(res_object >= match_threshold)
            for object_tr in zip(*loc_object[::-1]):
                object_coord = object_tr[0], object_tr[1]

        input = (dino_coord[0], dino_coord[1], object_coord[0], object_coord[1])

        fitness = int(round(time.time() - start_time))*10

        if driver.execute_script("return Runner.instance_.crashed") == True:
            listener.stop()
            driver.close()
            driver.quit()
            actions = moves
            print 'Fitness: ' + str(fitness)
            print 'Actions: ' + str(actions)
            moves = 0
            return fitness - actions

        output = net.activate(input)

        if output[0] > 0.5:
            keyboard.press(Key.space)
            time.sleep(0.5)
            keyboard.release(Key.space)
            listener.stop()
        else:
            pass
        if output[1] > 0.5:
            keyboard.press(Key.down)
            time.sleep(0.5)
            keyboard.release(Key.down)
            listener.stop()
        else:
            pass

        listener.stop()


def eval_genoms(genomes, config):

    i = 0

    global score
    global generation, max_fitness, best_genome

    generation += 1
    for genome_id, genome in genomes:

        genome.fitness = game(genome, config)
        print ("Gen : %d Genome # : %d  Fitness : %f Max Fitness : %f"%(generation,i,genome.fitness, max_fitness))
        if genome.fitness >= max_fitness:
            max_fitness = genome.fitness
            best_genome = genome
        score = 0
        i += 1

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config')

pop = neat.Population(config)
stats = neat.StatisticsReporter()
pop.add_reporter(stats)

winner = pop.run(eval_genoms, 10)

serialNo = len(os.listdir(home))+1
outputFile = open(str(serialNo)+'_'+str(int(max_fitness))+'.p', 'wb')
print outputFile

pickle.dump(winner,outputFile)