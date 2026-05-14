#!/usr/bin/env python

# =============================================================================
# Project: iAntGA - Genetic Algorithm for ARGoS3
# Author: Charles A. Galperin
# Updated: April 2026
# -----------------------------------------------------------------------------
# CORE ALGORITHMIC INTEGRITY:
# The fundamental logic governing population evolution, including selection,
# crossover, mutation, and convergence criteria, remains identical to the
# original implementation by Dr. Qi Lu.
# -----------------------------------------------------------------------------
# SUMMARY OF INFRASTRUCTURAL ENHANCEMENTS:
#
# 1. PARALLELIZATION:
#    Refactored the execution pipeline from sequential to multi-processed
#    using 'multiprocessing.Pool'. This allows concurrent simulation
#    evaluations, drastically reducing runtime while maintaining 1:1 logical
#    consistency with the original GA process.
#
# 2. STATE PERSISTENCE (CHECKPOINTING):
#    Introduced a robust serialization system using 'pickle'. The
#    GA now saves the full experimental state (fitness, population, and
#    evolutionary counters) into .pkl checkpoints, allowing for seamless
#    resumption of a generation following interruptions.
#
# 3. NON-EVOLUTIONARY EVALUATION MODE:
#    Added functionality to load specific parameters via CSV for "Final Exam"
#    validation trials. This uses the same parallel architecture to
#    run N independent trials on a fixed individual.
#
# 4. MODERNIZED PATH & FILE MANAGEMENT:
#    - Replaced legacy 'os' and 'errno' calls with 'pathlib' for cross-platform
#      reliability.
#    - Implemented a "Lost & Found" recovery system to safeguard data from
#      crashed runs.
#    - Standardized experiment archiving and timestamped logging.
#
# 5. BUG FIXES & STABILITY:
#    - Implemented a stable sorting algorithm to prevent 'TypeError' crashes
#      when identical fitness scores are compared.
#    - Fixed state 'amnesia' during checkpoint loads by ensuring all evolutionary
#      tracking variables (not_evolved_count, etc.) are correctly restored.
#
# 6. DOCUMENTATION & UI:
#    - Added PEP 257 docstrings to all major functions.
#    - Integrated 'tqdm' for real-time visual progress monitoring.
#    - Implemented 'print_time()' for structured console output.
# =============================================================================
import argos_util
import subprocess
import csv
import tempfile
import os
import numpy as np
import time
import argparse
import copy
from lxml import etree
import logging
import datetime
import pickle  # required for save/load feature
import sys  # used to interrupt program
import shutil  # manage file outputs from CPFA logs
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  #show progress
from pathlib import Path  # Updated path management


def print_time(*message_parts, **kwargs):
    """
    Modified print statement that adds a timestamp prefix. Treat it as you would a print().
    """
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}]", *message_parts, **kwargs)


def run_argos_simulation(task_data):
    """
    Executes an individual ARGoS3 simulation in a separate process.

    Args:
        task_data (tuple): A tuple containing (pop_idx, xml_str, seed, test_id).
            - pop_idx (int): The index of the population member.
            - xml_str (str): The raw XML content of the simulation configuration.
            - seed (int): The random seed for reproducibility.
            - test_id (int): The ID of the current evaluation test.

    Returns:
        float: The fitness value achieved by the robot controller.
    """
    pop_idx, xml_str, seed, test_id = task_data

    xml_obj = etree.fromstring(xml_str)
    argos_util.set_seed(xml_obj, seed)
    final_xml_str = etree.tostring(xml_obj)

    # Create the temp file locally within the worker process
    # Path.cwd() / "experiments" is the pathlib equivalent of os.path.join(os.getcwd(), "experiments")
    tmpf = tempfile.NamedTemporaryFile('wb', suffix=".argos", prefix="gatmp",
                                       dir=Path.cwd() / "experiments",
                                       delete=False)
    tmpf.write(final_xml_str)
    tmpf.close()

    # Start Log
    logging.info("pop %d at test %d with seed %d", pop_idx, test_id, seed)

    # Run the simulation
    argos_args = ["argos3", "-n", "-c", tmpf.name]
    argos_run = subprocess.run(argos_args, stdout=subprocess.PIPE, text=True)

    # Cleanup temp file
    tmp_path = Path(tmpf.name)
    if tmp_path.exists():
        tmp_path.unlink()

    if argos_run.returncode != 0:
        logging.error("Argos failed test")
        return 0.0

    # Extract fitness from output (assuming format "fitness, ...")
    output = argos_run.stdout.strip().split('\n')
    if not output: return 0.0

    # Get the last line of output
    # Split by newline and take the last element [-1]
    last_line = argos_run.stdout.strip().split('\n')[-1]
    parts = last_line.split(",")
    fitness_val = float(parts[0])
    sim_time = parts[1]
    sim_seed = parts[2]
    # Atomic Result Log (Includes pop and test IDs so it's never ambiguous)
    logging.info("partial fitness = %f for pop %d at test %d", fitness_val, pop_idx, test_id)
    tqdm.write(f"\nPopulation {pop_idx} Test {test_id}: {fitness_val}, {sim_time}, {sim_seed}")

    return fitness_val


class ArgosRunException(Exception):
    pass


class iAntGA(object):
    """
    Manages the Genetic Algorithm for optimizing ARGoS3 robot controllers.

    Handles population initialization, generation iteration, simulation
    parallelization, and checkpointing of experiment states.

    Attributes:
        xml_file (str): Path to the base ARGoS3 configuration file.
        population (list): List of lxml.etree elements representing robot parameters.
        save_dir (Path): Directory where experiment results are archived.
    """
    def __init__(self,
                 xml_file,
                 pop_size=50,
                 gens=20,
                 elites=3,
                 mut_rate=0.1,
                 robots=20,
                 tags=1024,
                 length=3600,
                 system="linux",
                 tests_per_gen=10,
                 terminateFlag=0,
                 resume_file=None,
                 num_workers=1,
                 skip_init=False):

        self.xml_file = xml_file  #qilu 03/26/2016
        self.system = system
        self.pop_size = pop_size
        self.gens = gens
        self.elites = elites
        self.mut_rate = mut_rate
        self.current_gen = 0
        self.robots = robots  #qilu 03/26/2016
        self.tags = tags
        # Initialize population
        self.population_data = []
        self.population = []
        self.prev_population = None
        self.system = system
        self.fitness = np.zeros(pop_size)
        self.start_time = int(time.time())
        self.length = length
        self.tests_per_gen = tests_per_gen
        self.terminateFlag = terminateFlag  #qilu 01/21/2016
        self.not_evolved_idx = [-1] * self.pop_size  #qilu 03/27/2016 check whether a population is from previous generation and is not modified
        self.not_evolved_count = [0] * self.pop_size  #qilu 04/02
        self.prev_not_evolved_count = [0] * self.pop_size  #qilu 04/02
        self.prev_fitness = np.zeros(pop_size)  #qilu 03/27/2016
        xml_experiment_file = Path(xml_file).stem

        # Directory for experiment
        dir_string = f"EVAL_{xml_experiment_file}_{self.start_time}_e_{elites}_p_{pop_size}_r_{robots}_tag_{tags}_t_{length}_k_{tests_per_gen}"
        # If True, skips randomly generating population and/or continued evolution
        if not skip_init:
            dir_string = f"{xml_experiment_file}_{self.start_time}_e_{elites}_p_{pop_size}_r_{robots}_tag_{tags}_t_{length}_k_{tests_per_gen}"
            if resume_file and Path(resume_file).exists():
                print_time(f"Resuming from {resume_file}...")
                self.load_state(resume_file)
            else:
                print_time("Starting a fresh simulation...\n")
                for _ in range(pop_size):
                    self.population.append(argos_util.uniform_rand_argos_xml(xml_file, robots, length, system))

        self.num_workers = num_workers
        # Save location for experiment output and logs
        self.save_dir = Path("gapy_saves") / dir_string
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # Save location for checkpoints within the save directory already used for the experiment
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(filename=self.save_dir / 'iAntGA.log',
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.DEBUG
                            )


    def save_state(self):
        """
        Serializes the current GA state to a pickle file.

        Converts internal lxml objects to strings (since lxml objects are not
        picklable) and saves the class instance to a .pkl file in the
        checkpoints directory.
        """
        # Convert lxml objects to xml strings for pickling since lxml cant be pickled
        xml_strings_pop = [etree.tostring(ind) for ind in self.population]
        xml_strings_prev = None
        if self.prev_population is not None:
            xml_strings_prev = [etree.tostring(ind) for ind in self.prev_population]

        # Temporarily replace xml data with string
        original_population = self.population
        original_prev_population = self.prev_population

        self.population = xml_strings_pop
        self.prev_population = xml_strings_prev

        # Create an identifiable checkpoint name
        # Use Path.stem to get the filename without extension automatically
        base_exp_name = Path(self.xml_file).stem
        checkpoint_file_name = f"{base_exp_name}_Gen_{self.current_gen}_of_{self.gens}.pkl"
        path_to_save = Path(self.checkpoint_dir) / checkpoint_file_name

        with open(path_to_save, 'wb') as f:
            pickle.dump(self, f)
        # restore with original lxml data before continuing experiment process
        self.population = original_population
        self.prev_population = original_prev_population
        print_time(f"State saved to {checkpoint_file_name}")


    def load_state(self, filename):
        """
        Loads and restores a GA state from a pickle checkpoint file.

        Args:
            filename (str or Path): The path to the .pkl checkpoint file.
        """
        with open(filename, 'rb') as f:
            loaded_ga = pickle.load(f)
            # Worker number safety check for older checkpoints
            if not hasattr(loaded_ga, 'num_workers'):
                print_time(f"Number of workers not found in file, initializing them...")
                loaded_ga.num_workers = 1  # A default value should it not exist

            # Update current instance attributes
            self.num_workers = loaded_ga.num_workers
            self.fitness = loaded_ga.fitness
            self.current_gen = loaded_ga.current_gen
            self.prev_fitness = loaded_ga.prev_fitness
            self.not_evolved_idx = loaded_ga.not_evolved_idx
            self.not_evolved_count = loaded_ga.not_evolved_count
            self.prev_not_evolved_count = loaded_ga.prev_not_evolved_count

            # These two are currently strings
            self.population = loaded_ga.population
            self.prev_population = loaded_ga.prev_population
            # Turn loaded string data back to lxml
            self.population = [etree.fromstring(xml_str) for xml_str in self.population]
            # Safety check for previous population
            prev_pop_raw = getattr(loaded_ga, 'prev_population', None)
            if prev_pop_raw is not None:
                self.prev_population = [etree.fromstring(xml_str) for xml_str in self.prev_population]
            else:
                self.prev_population = None
            print_time(f"Resumed from {filename} at generation {self.current_gen}\n")


    def archive_results(self):
        """
        Moves completed log files from the temporary 'results/' directory
        to the final experiment save directory.
        """
        results_dir = Path.cwd() / "results"

        if results_dir.exists():
            save_path = Path(self.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            # .glob("*") gets all files; .suffix checks extension
            for file_path in results_dir.glob("*"):
                if file_path.suffix in [".txt", ".csv"]:
                    shutil.move(str(file_path), str(save_path / file_path.name))
                    logging.info(f"Archived {file_path.name} to {save_path}")


    def recover_lost_files(self):
        """
        Moves leftover log files that may exist from a previous incomplete run do to a crash to the
        'experiments/_lostandfound/' directory before a new experiment deletes them.
        """
        results_dir = Path.cwd() / "results"
        lost_found_dir = Path.cwd() / "gapy_saves" / "_lostandfound"

        if results_dir.exists() and any(results_dir.iterdir()):
            print_time("Found leftover files in results/ folder. Moving them to gapy_saves/_lostandfound...")
            lost_found_dir.mkdir(parents=True, exist_ok=True)

            for file_path in results_dir.glob("*"):
                if file_path.suffix in [".txt", ".csv"]:
                    shutil.move(str(file_path), str(lost_found_dir / file_path.name))
                    logging.info(f"Recovered {file_path.name} to {lost_found_dir}")


    def run_ga(self):
        """
        Executes a single generation of the GA.

        1. Prepares XML tasks for the worker pool.
        2. Executes simulations in parallel using multiprocessing.
        3. Aggregates fitness scores and performs crossover/mutation.
        4. Saves state and checkpoints.
        """
        results_dir = Path.cwd() / "results"
        self.recover_lost_files()
        results_dir.mkdir(exist_ok=True)
        print_time("num_workers = " + str(num_workers))
        print_time("pop_size = " + str(pop_size))
        print_time("gens = " + str(gens))
        print_time("elites = " + str(elites))
        print_time("mut_rate = " + str(mut_rate))
        print_time("robots = " + str(robots))
        print_time("tags = " + str(tags))
        print_time("time = " + str(length / 60) + " minutes")
        print_time("evaluations = " + str(tests_per_gen))
        print()
        # New execution phase to archive files instead of only doing so
        # when a full run is completed
        try:
            while self.current_gen <= self.gens and self.terminateFlag == 0:
                self.run_generation()
        except KeyboardInterrupt:
            print_time("Simulation interrupted (Ctrl+C). Cleaning up...")
        finally:
            # This ALWAYS runs: archives new files to experiment folder,
            # cleans results/ folder for next time.
            self.archive_results()
            print_time("Cleanup complete. Results saved.")


    def run_generation(self):
        """
        Executes a single generation of the Genetic Algorithm.

        This method manages the full generational lifecycle:
        1. Initializes fitness arrays and calculates fitness in parallel
            using the worker pool.
        2. Sorts the population based on fitness and preserves elites.
        3. Performs crossover and mutation to generate the new offspring pool.
        4. Increments the generation counter and checkpoints the state.
        """
        logging.info("Starting generation: " + str(self.current_gen))
        print_time(f"--- Starting Generation {self.current_gen} ---")
        self.fitness = np.zeros(pop_size)  #reset it
        seeds = [np.random.randint(2 ** 32) for _ in range(self.tests_per_gen)]
        logging.info("Seeds for generation: " + str(seeds))


        # Prepare tasks for the pool
        xml_strings = [etree.tostring(p) for p in self.population]
        tasks = []
        task_map = []  # To keep track of which result belongs to which population index

        for i in range(self.pop_size):
            if self.not_evolved_idx[i] == -1 or self.not_evolved_count[i] > 3:
                self.not_evolved_count[i] = 0
                for test_id, seed in enumerate(seeds):
                    tasks.append((i, xml_strings[i], seed, test_id))
                    task_map.append(i)  # Log that this task belongs to population 'i'
            else:
                self.fitness[i] = self.prev_fitness[self.not_evolved_idx[i]] * self.tests_per_gen

        # Run in parallel
        if tasks:
            # imap() instead of map() to get an iterator
            # tqdm wraps that iterator to create the progress bar
            with Pool(processes=self.num_workers) as pool:
                results = []
                for result in tqdm(pool.imap(run_argos_simulation, tasks),
                                   total=len(tasks),  # tells the bar how many steps to expect
                                   desc=f"Generation {self.current_gen}",
                                   ncols=64,  # sets the size of the bar
                                   position=0,  #bar stays at the bottom
                                   leave=True):
                    results.append(result)

            # Aggregate results
            for result_idx, score in enumerate(results):
                pop_index = task_map[result_idx]
                self.fitness[pop_index] += score

        # use average fitness as fitness
        for i in range(len(self.fitness)):
            logging.info("pop %d total fitness = %g", i, self.fitness[i])
            self.fitness[i] /= self.tests_per_gen
            logging.info("pop %d avg fitness = %g", i, self.fitness[i])

        # sort fitness and population
        fit_pop_index = range(len(self.fitness))
        # Sort based on self.fitness only
        sorted_fit_pop_index = sorted(fit_pop_index, key=lambda x: self.fitness[x], reverse=True)
        # rebuild list after sorting
        fit_pop = [(self.fitness[i], self.population[i], self.not_evolved_count[i]) for i in sorted_fit_pop_index]

        self.fitness, self.population, self.not_evolved_count = map(list, zip(*fit_pop))

        self.save_population(seed)

        self.prev_population = copy.deepcopy(self.population)
        self.prev_fitness = copy.deepcopy(self.fitness)  #qilu 03/27
        self.prev_not_evolved_count = copy.deepcopy(self.not_evolved_count)  #qilu 04/02

        self.not_evolved_idx = []  #qilu 03/27/2016
        self.not_evolved_count = []  #qilu 04/02/2016
        self.population = []
        self.check_termination()  #qilu 01/21/2016 add this function
        self.population_data = []  # qilu 01/21/2016 reset it
        # Add elites
        for i in range(self.elites):
            # reverse order from sort
            self.population.append(self.prev_population[i])
            self.not_evolved_idx.append(i)
            self.not_evolved_count.append(self.prev_not_evolved_count[i] + 1)

        # Now do crossover and mutation until population is full

        num_newOffSpring = self.pop_size - self.elites
        #pdb.set_trace()
        count = 0
        for i in range(num_newOffSpring):
            if count == num_newOffSpring: break
            p1c = np.random.choice(len(self.prev_population), 2)
            p2c = np.random.choice(len(self.prev_population), 2)
            if p1c[0] <= p1c[1]:
                parent1 = self.prev_population[p1c[0]]
                idx1 = p1c[0]
            else:
                parent1 = self.prev_population[p1c[1]]
                idx1 = p1c[1]

            if p2c[0] <= p2c[1]:
                parent2 = self.prev_population[p2c[0]]
                idx2 = p2c[0]
            else:
                parent2 = self.prev_population[p2c[1]]
                idx2 = p2c[1]
            #if parent1 != parent2 and np.random.uniform()<0.5: #qilu 11/26/2015
            #pdb.set_trace()
            if parent1 != parent2:  #qilu 03/26/2016
                children = argos_util.uniform_crossover(xml_file, parent1, parent2, 0.5,
                                                        self.system)  # qilu 03/07/2016 add the crossover rate p
            else:
                children = [copy.deepcopy(parent1), copy.deepcopy(parent2)]
            for child in children:
                argos_util.mutate_parameters(child, self.mut_rate)
                self.population.append(child)
                if argos_util.get_parameters(parent1) == argos_util.get_parameters(child):
                    #pdb.set_trace()
                    self.not_evolved_idx.append(idx1)
                    self.not_evolved_count.append(self.prev_not_evolved_count[idx1] + 1)
                elif argos_util.get_parameters(parent2) == argos_util.get_parameters(child):
                    #pdb.set_trace()
                    self.not_evolved_idx.append(idx2)
                    self.not_evolved_count.append(self.prev_not_evolved_count[idx2] + 1)
                else:
                    self.not_evolved_idx.append(-1)
                    self.not_evolved_count.append(0)
            count += 2
            while count > num_newOffSpring:
                del self.population[-1]
                del self.not_evolved_idx[-1]
                del self.not_evolved_count[-1]
                count -= 1
        self.current_gen += 1
        # (4/19/2026) Charles Galperin
        # Checkpoint created after the computation of a generation
        self.save_state()


    def check_termination(self):
        """
        Evaluates the termination criteria for the GA.

        Determines if the evolutionary process should stop, typically
        checked against the generation limit or convergence criteria.

        Returns:
            bool: True if termination conditions are met, False otherwise.
        """
        upperBounds = [1.0, 1.0, 2.0, 20.0, 1.0, 20.0, 180.0]
        fitness_convergence_rate = 0.95
        diversity_rate = 0.035
        data_keys = sorted(self.population_data[0].keys())  # (4/19/2026) Charles Galperin: Python3 fix
        complete_data = []
        for data in self.population_data:
            complete_data.append([float(data[key]) for key in data_keys])
        npdata = np.array(complete_data)

        #Fitness convergence and population diversity
        means = npdata.mean(axis=0)
        stds = np.delete(npdata.std(axis=0), [7, 8])
        #pdb.set_trace()
        normalized_stds = stds / upperBounds

        current_fitness_rate = means[7] / npdata[0, 7]
        current_diversity_rate = normalized_stds.max()
        if current_diversity_rate <= diversity_rate and current_fitness_rate >= fitness_convergence_rate:
            self.terminateFlag = 1
            print_time("Convergent...\n")
        elif current_diversity_rate > diversity_rate and current_fitness_rate < fitness_convergence_rate:
            print_time('Fitness is not convergent...')
            print_time('Fitness rate is ' + str(current_fitness_rate))
            print_time('Deviation is ' + str(current_diversity_rate))
        elif current_diversity_rate > diversity_rate:
            print_time('population diversity is high...')
            print_time(
                'The current standard deviation is ' + str(current_diversity_rate) + ', which is greater than ' + str(
                    diversity_rate) + '...')
        else:
            print_time('Fitness is not convergent...')
            print_time(
                'The current rate of mean of fitness is ' + str(current_fitness_rate) + ', which is less than ' + str(
                    fitness_convergence_rate) + '...')


    def save_population(self, seed):
        """
        Persists the current population and fitness data to disk.

        Args:
            seed (int or str): The random seed associated with the current
                simulation run; used to identify the saved data file
                or log entry.
        """
        save_path = Path(self.save_dir)
        save_path.mkdir(exist_ok=True)
        filename = f"gen_{self.current_gen}.gapy"
        #population_data = []
        for f, p in zip(self.fitness, self.population):
            data = copy.deepcopy(argos_util.get_parameters(p))
            #data2= copy.deepcopy(argos_util.get_controller_params(p)) #qilu 07/25
            if 'PrintFinalScore' in data:
                del data['PrintFinalScore']
            data["fitness"] = f
            data["seed"] = seed
            self.population_data.append(data)
            #population_data2.append(data2)
            #print data
        # (4/19/2026) Charles Galperin
        # Python3 fix
        data_keys = sorted(list(argos_util.PARAMETER_LIMITS.keys()) + ["fitness", "seed"])

        #data_keys2 = argos_util.controller_params_LIMITS.keys()
        #data_keys2.sort()
        with open(save_path / filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data_keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.population_data)  #qilu 07/27
            #writer2 = csv.DictWriter(csvfile, fieldnames=data_keys2, extrasaction='ignore')
            #writer2.writeheader()
            #writer2.writerows(population_data2) #qilu 07/27


    def evaluate_best(self, best_xml_str, num_trials=50):
        """
        Runs a "Final Exam" for a specific controller configuration.
        Bypasses evolution and runs the configuration through N simulations.
        """
        print_time(f"Starting Evaluation Mode: {num_trials} trials.")

        # Prepare the tasks (same logic as run_generation)
        seeds = [np.random.randint(2 ** 32) for _ in range(num_trials)]

        # Fill the population with the same "best" individual
        tasks = []
        for test_id, seed in enumerate(seeds):
            tasks.append((0, best_xml_str, seed, test_id))

        # Run the pool
        with Pool(processes=self.num_workers) as pool:
            results = list(tqdm(pool.imap(run_argos_simulation, tasks),
                                total=len(tasks),
                                desc="Evaluation Trials"))

        # Calculate results
        avg_fitness = sum(results) / len(results)
        print_time(f"Evaluation Complete. Average Fitness: {avg_fitness}")
        return avg_fitness


    def run_evaluation_from_csv(self, csv_file_path, base_xml_path, num_trials=10):
        """
        Reads best parameters from a CSV and runs them through the evaluation pipeline.
        CSV should be two rows that look like:
            -header0, header1, ..., headerN

            -value0, value1, ..., valueN
        """
        # Load the best parameters from the CSV
        with open(csv_file_path, 'r') as f:
            reader = csv.DictReader(f)
            best_params = next(reader)  # Takes the first row (the best one)

        # Filter the dictionary: only keep keys that exist in PARAMETER_LIMITS
        # This prevents 'fitness' or 'seed' from being injected into the XML
        from argos_util import PARAMETER_LIMITS
        valid_params = {k: v for k, v in best_params.items() if k in PARAMETER_LIMITS}

        # Create the base XML using existing util
        xml = argos_util.default_argos_xml(base_xml_path, time=self.length, system=self.system)

        # Inject!
        argos_util.set_parameters(xml, valid_params)
        xml_str = etree.tostring(xml).decode('utf-8')

        # Run the evaluation using existing parallel pipeline
        print_time("Evaluating parameters:")
        for key, value in valid_params.items():
            print_time(f"  {key}: {value}")
        self.evaluate_best(xml_str, num_trials=num_trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GA for argos')
    parser.add_argument('-f', '--file', action='store', dest='xml_file')
    parser.add_argument('-s', '--system', action='store', dest='system')
    parser.add_argument('-r', '--robots', action='store', dest='robots', type=int)
    parser.add_argument('-m', '--mut_rate', action='store', dest='mut_rate', type=float)
    parser.add_argument('-e', '--elites', action='store', dest='elites', type=int)
    parser.add_argument('-g', '--gens', action='store', dest='gens', type=int)
    parser.add_argument('-p', '--pop_size', action='store', dest='pop_size', type=int)
    parser.add_argument('-t', '--time', action='store', dest='time', type=int)
    parser.add_argument('-k', '--tests_per_gen', action='store', dest='tests_per_gen', type=int)
    parser.add_argument('-o', '--terminateFlag', action='store', dest='terminateFlag', type=int)
    # (4/19/2026) Charles Galperin
    # flag added for loading checkpoint files
    parser.add_argument('-rf', '--resume_file', action='store', dest='resume_file',
                        help='Path to a .pkl checkpoint file')
    # (4/24/2026) Charles Galperin
    parser.add_argument("-w", "--workers", action='store', dest='num_workers', default=os.cpu_count() or 4,
                        help="Number of parallel workers (default: # of CPUs)", type=int)
    # (4/25/2026) Charles Galperin
    parser.add_argument('-ev','--eval-csv', action='store', dest='eval_csv',
                        help="CSV file with best parameters")
    parser.add_argument('-evt','--trials', action='store', default=10,
                        help="Number of evaluation trials", type=int)

    pop_size = 50
    gens = 100
    elites = 1
    mut_rate = 0.05
    robots = 24  #robots = 16
    tags = 256  #qilu 03/26 for naming the output directory
    system = "linux"
    length = 720  # 12 minutes, length is in second. default length = 3600
    tests_per_gen = 10
    terminateFlag = 0

    args = parser.parse_args()

    #xml_file = raw_input('Choose a file name(e.g. cluster_2_mac.argos)')
    if args.xml_file:
        xml_file = args.xml_file
        print_time("The input file: " + xml_file)

    if args.pop_size:
        pop_size = args.pop_size

    if args.gens:
        gens = args.gens

    if args.elites:
        elites = args.elites

    if args.mut_rate:
        mut_rate = args.mut_rate

    if args.robots:
        robots = args.robots

    if args.system:
        system = args.system

    if args.time:
        length = args.time

    if args.tests_per_gen:
        tests_per_gen = args.tests_per_gen

    if args.terminateFlag:
        terminateFlag = args.terminateFlag

    # Logic for loading a checkpoint or starting a fresh experiment
    if args.resume_file:
        resume_path = Path(args.resume_file)  # Convert to Path object once
        if not resume_path.exists():
            print(f"ERROR: Resume file '{args.resume_file}' not found.")
            # Stop the script entirely with an error code
            sys.exit(1)
        else:
            resume_file = args.resume_file
    else:
        resume_file = None

    if args.num_workers:
        num_workers = args.num_workers

    start = time.time()

    # Evaluation mode if a CSV is loaded with parameters to test
    if args.eval_csv:
        eval_csv_path = Path(args.eval_csv)
        # Safety Check: If no CSV is found, do not continue.
        if not eval_csv_path.exists():
            print_time(f"ERROR: Evaluation CSV file '{args.eval_csv}' not found.")
            sys.exit(1)

        if args.trials:
            trials = args.trials

        eval_csv_file = args.eval_csv
        ga = iAntGA(xml_file=xml_file,
                    pop_size=pop_size,
                    gens=gens,
                    elites=elites,
                    mut_rate=mut_rate,
                    robots=robots,
                    tags=tags,
                    length=length,
                    system=system,
                    tests_per_gen=tests_per_gen,
                    terminateFlag=terminateFlag,
                    resume_file=resume_file,
                    num_workers=num_workers,
                    skip_init=True,
                    )
        ga.run_evaluation_from_csv(eval_csv_file, xml_file, trials)
        ga.archive_results()

    # Otherwise original GA evolutionary behavior
    else:
        ga = iAntGA(xml_file=xml_file,
                    pop_size=pop_size,
                    gens=gens,
                    elites=elites,
                    mut_rate=mut_rate,
                    robots=robots,
                    tags=tags,
                    length=length,
                    system=system,
                    tests_per_gen=tests_per_gen,
                    terminateFlag=terminateFlag,
                    resume_file=resume_file,
                    num_workers=num_workers,
                    skip_init=False
                    )
        ga.run_ga()

    stop = time.time()
    duration = int(stop - start)
    readable_time = str(datetime.timedelta(seconds=duration))
    print_time('The loaded file is ' + xml_file)
    print_time(f'Experiment runtime: {readable_time}')
