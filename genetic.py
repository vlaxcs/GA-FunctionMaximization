import json
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
            coefficients: list[int] = [], 
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

        interval_length = upper_bound - lower_bound
        necessary_values = interval_length * (10 ** precision)
        self.chromosome_length = math.ceil(math.log2(necessary_values))


    def f(self, x):
        result = self.coefficients[0]

        # Adaptation for any polynomial - Horner
        for idx in range(1, len(self.coefficients)):
            result = result * x + self.coefficients[idx]

        return result
    

    @lru_cache(maxsize = None)
    def decodeChromosome(self, chromosome):
        value = int(chromosome, 2)
        return self.lower_bound + value * (self.upper_bound - self.lower_bound) / ((2 ** self.chromosome_length) - 1)
    

    def generateChromosome(self):
        return ''.join(random.choice(['0', '1'] for _ in range(self.chromosome_length)))
    

    def binarySearch(self, intervals, target):
        left, right = 0, len(intervals) - 1

        while left <= right:
            mid = (left + right) >> 1

            if intervals[mid] <= target < intervals[mid + 1]:
                return mid
            elif target < intervals[mid]:
                right = mid - 1
            else:
                left = mid + 1
            
        return len(intervals) - 2
    

    def buildIntervals(self, fitness, total_fitness):
        proabilities = [fit / total_fitness for fit in fitness]
        interval_lower_bounds = [0.0]

        for p in proabilities:
            interval_lower_bounds.append(interval_lower_bounds[-1] + p)

        interval_lower_bounds[-1] = 1.0

        return interval_lower_bounds
    
    
    def generateIntervals(self, intervals, population: list[str]):
        new_population = []

        for _ in range(self.population_size):
            new = random.random()
            interval_index = self.binarySearch(intervals, new)
            new_population.append(population[interval_index])

        return new_population
            
    @staticmethod
    def log(container: list[str], repr):
        container.append(repr)


    def run(self):
        population = [self.generateChromosome() for _ in range(self.population_size)]
        max_history = []
        mean_history = []

        # mai am de făcut loggingu pe prima generatie
        for generation in range(self.generation_count):
            x_values = [self.decodeChromosome(chromosome) for chromosome in population]
            fitness = [self.f(x) for x in x_values]
            max_fitness = max(fitness)
            total_fitness = sum(fitness)
            mean_fitness = total_fitness / self.population_size            
            elite_chromosome = population[fitness.index(max_fitness)]

            intervals = self.buildIntervals(fitness, total_fitness)
            new_population = self.generateIntervals(intervals, population)

            # Crossover
            parents_idx = [idx for idx in range(self.population_size) if random.random() < self.crossover_probability]
            random.shuffle(parents_idx)

            for idx in range(0, len(parents_idx) - 1, 2):
                split_point = random.randint(0, self.chromosome_length - 1)
                
                p1_idx, p2_idx = parents_idx[idx], parents_idx[idx + 1]
                p1, p2 = new_population[p1_idx], new_population[p2_idx]
                new_population[p1_idx] = p1[:split_point] + p2[split_point:]
                new_population[p2_idx] = p2[:split_point] + p1[split_point:]
        

            # Mutation
            for might in range(self.population_size):
                chromosome = list(new_population[might])

                for happen in range(self.chromosome_length):
                    if random.randint() < self.mutation_probability:
                        chromosome[happen] = '1' if chromosome[happen] == '0' else '0'
                
                new_population[might] = ''.join(chromosome)

            new_fitness = [self.f(self.decodeChromosome(chromosome)) for chromosome in new_population]
            new_population[new_fitness.index(min(new_fitness))] = elite_chromosome
            population = new_population

        self.log(max_history, fitness)  # will replace with logging into app instance
        self.log(mean_history, mean_fitness)

        return max_history, mean_history