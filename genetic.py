# Non-generated code

import math
import random
from functools import lru_cache

class GeneticAlgorithm:
    def __init__(
        self,
        population_size,
        crossover_probability,
        mutation_probability,
        generation_count,
        generation_rate,
        steps,
        precision,
        lower_bound,
        upper_bound,
        coefficients: list[float] = [],
    ):
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.generation_rate = generation_rate
        self.generation_count = generation_count
        self.steps = steps
        self.precision = precision
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.coefficients = coefficients

        self.probabilities_log = str()

        interval_length = upper_bound - lower_bound
        necessary_values = max(interval_length * (10 ** precision), 2)
        self.chromosome_length = math.ceil(math.log2(necessary_values))

    def f(self, x):
        result = self.coefficients[0]

        # Adaptation for any polynomial - Horner
        for idx in range(1, len(self.coefficients)):
            result = result * x + self.coefficients[idx]

        return result

    @lru_cache(maxsize=None)
    def decodeChromosome(self, chromosome):
        value = int(chromosome, 2)
        return self.lower_bound + value * (self.upper_bound - self.lower_bound) / ((2 ** self.chromosome_length) - 1)

    def generateChromosome(self):
        return ''.join(random.choice(['0', '1']) for _ in range(self.chromosome_length))

    def binarySearch(self, intervals, target):
        left, right = 0, len(intervals) - 2

        while left <= right:
            mid = (left + right) >> 1

            if intervals[mid] <= target < intervals[mid + 1]:
                return mid
            if target < intervals[mid]:
                right = mid - 1
            else:
                left = mid + 1

        return len(intervals) - 2

    def buildIntervals(self, fitness, total_fitness, should_log = False):
        if total_fitness <= 0:
            probability = 1.0 / len(fitness)
            probabilities = [probability for _ in fitness]
        else:
            probabilities = [fit / total_fitness for fit in fitness]

            if should_log:
                self.probabilities_log = probabilities

        interval_lower_bounds = [0.0]

        for probability in probabilities:
            interval_lower_bounds.append(interval_lower_bounds[-1] + probability)

        interval_lower_bounds[-1] = 1.0

        return interval_lower_bounds

    def generateIntervals(self, intervals, population: list[str]):
        new_population = []

        for _ in range(self.population_size):
            value = random.random()
            interval_index = self.binarySearch(intervals, value)
            new_population.append(population[interval_index])

        return new_population

    @staticmethod
    def log(container: list[str], value):
        container.append(value)

    @staticmethod
    def writeExport(population, x_values, fitness, probabilities, intervals):
        with open('Evolutie.txt', 'w') as f:
            f.write('Populatia initiala\n')
            for idx, (chromosome, x, f_x) in enumerate(zip(population, x_values, fitness)):
                f.write(f'\t{str(idx + 1).rjust(3)}: {chromosome}\t {str('x = ' + str(x)).ljust(30)}\t{str('f = ' + str(f_x)).ljust(30)}\n')

            f.write('\nProbabilitati selectie\n')
            for idx, (chromosome, probability) in enumerate(zip(population, probabilities)):
                f.write(f'\t{str(idx + 1).rjust(3)}: {chromosome}\t probabilitiate = {probability}\n')
            
            f.write('\nIntervale probabilitati selectie\n')
            for interval in intervals:
                f.write(f'\t{interval}\n')

    @staticmethod
    def appendToExport(max_history):
        with open('Evolutie.txt', 'a') as f:
            f.write('\nEvolutia maximului\n')
            for fitness in max_history:
                f.write(f'\t{fitness}\n')

    def run(self):
        population = [self.generateChromosome() for _ in range(self.population_size)]
        max_history = []
        mean_history = []

        for generation_idx in range(self.generation_count):
            x_values = [self.decodeChromosome(chromosome) for chromosome in population]
            fitness = [self.f(x) for x in x_values]
            max_fitness = max(fitness)
            mean_fitness = sum(fitness) / self.population_size
            elite_chromosome = population[fitness.index(max_fitness)]

            min_fitness = min(fitness)
            adjusted_fitness = [fit - min_fitness + 1e-12 for fit in fitness] if min_fitness <= 0 else fitness
            total_fitness = sum(adjusted_fitness)

            intervals = self.buildIntervals(adjusted_fitness, total_fitness, generation_idx == 0)
            new_population = self.generateIntervals(intervals, population)

            # Crossover
            parents_idx = [
                idx for idx in range(self.population_size) \
                if random.random() < self.crossover_probability
            ]
            random.shuffle(parents_idx)

            for idx in range(0, len(parents_idx) - 1, 2):
                split_point = random.randint(0, self.chromosome_length - 1)

                p1_idx, p2_idx = parents_idx[idx], parents_idx[idx + 1]
                p1, p2 = new_population[p1_idx], new_population[p2_idx]
                new_population[p1_idx] = p1[:split_point] + p2[split_point:]
                new_population[p2_idx] = p2[:split_point] + p1[split_point:]

            # Mutation
            for idx in range(self.population_size):
                chromosome = list(new_population[idx])

                for bit in range(self.chromosome_length):
                    if random.random() < self.mutation_probability:
                        chromosome[bit] = '1' if chromosome[bit] == '0' else '0'

                new_population[idx] = ''.join(chromosome)

            if generation_idx == 0:
                self.writeExport(population, x_values, fitness, self.probabilities_log, intervals)

            new_fitness = [self.f(self.decodeChromosome(chromosome)) for chromosome in new_population]
            new_population[new_fitness.index(min(new_fitness))] = elite_chromosome
            population = new_population

            self.log(max_history, max_fitness)
            self.log(mean_history, mean_fitness)
        
        self.appendToExport(max_history)
        return max_history, mean_history
