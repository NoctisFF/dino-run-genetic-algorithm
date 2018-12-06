import neat
import time
import pickle
import glob
import os
import cv2
import numpy as np
from PIL import ImageGrab
from selenium import webdriver
from pynput.keyboard import Key, Controller, Listener

home = os.path.dirname(os.path.abspath(__file__))
match_threshold = 0.8

generation = 0
max_fitness = 0
best_genome = 0

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
    Controller.release(Key.down)

def on_release(key):
    global moves
    moves += 1
    return False

def game(genome, config):

    global moves

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    driver = webdriver.Chrome()
    driver.implicitly_wait(10)
    driver.set_window_size(240, 320)
    driver.get("chrome://dino/")
    keyboard = Controller()
    keyboard.press(Key.space)

    time.sleep(2)

    global object_coord
    global dino_coord
    global delay
    max_speed = 100
    velocity_coeff = 1
    speed = 0
    delay = time.time()
    sec = time.time()

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

        if time.time() - sec >= 1 and time.time() - start_time < max_speed:
            speed += velocity_coeff
            sec = time.time()

        input = (object_coord[0] - dino_coord[0], dino_coord[1], object_coord[1], speed)

        distance = int(round(time.time() - start_time))*10

        if driver.execute_script("return Runner.instance_.crashed") == True:
            if listener.is_alive() == True:
                listener.stop()
            driver.close()
            driver.quit()
            actions = moves
            print 'Distance: ' + str(distance)
            print 'Actions: ' + str(actions)
            moves = 0
            return distance # - actions

        output = net.activate(input)
        print output[0]

        if output[0] > 0.9:
            keyboard.press(Key.space)
            delay = time.time()
        elif output[0] < 0.1:
            keyboard.press(Key.down)
            delay = time.time()
        else:
            pass

def eval_genoms(genomes, config):

    global max_fitness, generation, best_genome
    generation += 1

    for genome_id, genome in genomes:

        genome.fitness = game(genome, config)
        print ("Gen : %d Genome # : %d  Fitness : %f Max Fitness : %f"%(generation,genome_id,genome.fitness, max_fitness))
        if genome.fitness >= max_fitness:
            max_fitness = genome.fitness
            best_genome = genome

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config')

pop = neat.Population(config)
stats = neat.StatisticsReporter()
pop.add_reporter(stats)

winner = pop.run(eval_genoms, 40)

serialNo = len(os.listdir(home))+1
outputFile = open(str(serialNo)+'_'+str(int(max_fitness))+'.p', 'wb')
print outputFile

pickle.dump(winner,outputFile)
