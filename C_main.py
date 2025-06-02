import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
from typing import Tuple, List
import time

class GeneticImageEvolution:
    def __init__(self, image_path: str, image_size: Tuple[int, int] = (100, 100), population_size: int = 50):
        self.image_path = image_path
        self.width, self.height = image_size
        self.population_size = population_size
        self.pixel_count = self.width * self.height
        self.mutation_rate = 0.02
        self.crossover_rate = 0.8
        self.tournament_size = 5
        self.color_target = None
        self.grayscale_target = None
        self.binary_target = None
        
        self.fitness_history = {'color': [], 'grayscale': [], 'binary': []}
        
    def create_original_image(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path)
        img = img.resize((self.width, self.height))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    
    def rgb_to_grayscale(self, color_image: np.ndarray) -> np.ndarray:

        return np.dot(color_image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    
    def grayscale_to_binary(self, grayscale_image: np.ndarray, threshold: int = 128) -> np.ndarray:

        return ((grayscale_image > threshold) * 255).astype(np.uint8)
    
    def prepare_target_images(self):
    
        self.color_target = self.create_original_image(self.image_path)
    
        self.grayscale_target = self.rgb_to_grayscale(self.color_target)
    
        self.binary_target = self.grayscale_to_binary(self.grayscale_target)
    
    
    def create_random_chromosome(self, image_type: str) -> np.ndarray:
        if image_type == 'color':
            return np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
        elif image_type == 'grayscale':
            return np.random.randint(0, 256, (self.height, self.width), dtype=np.uint8)
        else:  # binary
            return (np.random.rand(self.height, self.width) > 0.5).astype(np.uint8) * 255
    
    def calculate_fitness(self, chromosome: np.ndarray, target: np.ndarray) -> float:
        mse = np.mean((chromosome.astype(float) - target.astype(float)) ** 2)
        max_possible_error = 255 ** 2
        fitness = max_possible_error - mse  # Higher fitness is better
        return fitness
    
    def tournament_selection(self, population: List[np.ndarray], target: np.ndarray) -> np.ndarray:
        tournament = random.sample(population, self.tournament_size)
        fitnesses = [self.calculate_fitness(chrom, target) for chrom in tournament]
        winner_idx = np.argmax(fitnesses)
        return tournament[winner_idx]
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        mask = np.random.rand(*parent1.shape) < 0.5
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        child1[mask] = parent2[mask]
        child2[mask] = parent1[mask]
        
        return child1, child2
    
    def mutate(self, chromosome: np.ndarray, image_type: str) -> np.ndarray:
        mutated = chromosome.copy()
        mutation_mask = np.random.rand(*chromosome.shape) < self.mutation_rate
        
        if image_type == 'color':
            mutated[mutation_mask] = np.random.randint(0, 256, size=np.sum(mutation_mask))
        elif image_type == 'grayscale':
            mutated[mutation_mask] = np.random.randint(0, 256, size=np.sum(mutation_mask))
        else:  # binary
            mutated[mutation_mask] = (np.random.rand(np.sum(mutation_mask)) > 0.5) * 255
        
        return mutated
    
    def evolve_image(self, image_type: str, max_generations: int = 5000, 
                    display_interval: int = 500, fitness_interval: int = 100) -> Tuple[np.ndarray, List[float]]:
        
        if image_type == 'color':
            target = self.color_target
        elif image_type == 'grayscale':
            target = self.grayscale_target
        else:
            target = self.binary_target
        
        population = [self.create_random_chromosome(image_type) for _ in range(self.population_size)]
        
        fitness_history = []
        best_chromosome = None
        best_fitness = float('-inf')
        
        start_time = time.time()
        
        for generation in range(max_generations):
            fitnesses = [self.calculate_fitness(chrom, target) for chrom in population]
            # Track best chromosome
            current_best_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_chromosome = population[current_best_idx].copy()
            
            if generation % fitness_interval == 0:
                avg_fitness = np.mean(fitnesses)
                fitness_history.append(avg_fitness)
                print(f"Generation {generation}: Avg Fitness = {avg_fitness:.2f}, Best Fitness = {best_fitness:.2f}")
            
            if generation % display_interval == 0 and generation > 0:
                self.display_chromosome(best_chromosome, f"{image_type.capitalize()} - Generation {generation}")
            
            new_population = []
            new_population.append(best_chromosome.copy())
            
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, target)
                parent2 = self.tournament_selection(population, target)
                
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutate(child1, image_type)
                child2 = self.mutate(child2, image_type)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        elapsed_time = time.time() - start_time
        print(f"Evolution completed in {elapsed_time:.2f} seconds")
        print(f"Final best fitness: {best_fitness:.2f}")
        
        return best_chromosome, fitness_history
    
    def display_chromosome(self, chromosome: np.ndarray, title: str):

        plt.figure(figsize=(4, 4))
        if len(chromosome.shape) == 3:  # Color image
            plt.imshow(chromosome)
        else:  # Grayscale or binary
            plt.imshow(chromosome, cmap='gray', vmin=0, vmax=255)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def display_all_images(self):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(self.color_target)
        axes[0].set_title('Original Color Image')
        axes[0].axis('off')
        
        axes[1].imshow(self.grayscale_target, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title('Grayscale Image')
        axes[1].axis('off')
        
        axes[2].imshow(self.binary_target, cmap='gray', vmin=0, vmax=255)
        axes[2].set_title('Binary Image')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_fitness_comparison(self):
        plt.figure(figsize=(12, 8))
        
        for image_type, history in self.fitness_history.items():
            if history:
                generations = range(0, len(history) * 100, 100)
                plt.plot(generations, history, label=f'{image_type.capitalize()} Image', linewidth=2)
        
        plt.xlabel('Generation')
        plt.ylabel('Average Fitness')
        plt.title('Fitness Evolution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def run_complete_evolution(self):
   
        self.prepare_target_images()
        self.display_all_images()
        
        max_generations = 5000
        display_interval = 1000
        fitness_interval = 500
        
        image_types = ['color', 'grayscale', 'binary']
        best_chromosomes = {}
        
        for image_type in image_types:
            best_chrom, fitness_hist = self.evolve_image(
                image_type, max_generations, display_interval, fitness_interval
            )
            best_chromosomes[image_type] = best_chrom
            self.fitness_history[image_type] = fitness_hist
        
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(self.color_target)
        axes[0, 0].set_title('Original Color')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(self.grayscale_target, cmap='gray', vmin=0, vmax=255)
        axes[0, 1].set_title('Original Grayscale')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(self.binary_target, cmap='gray', vmin=0, vmax=255)
        axes[0, 2].set_title('Original Binary')
        axes[0, 2].axis('off')
        
        # Evolved images
        axes[1, 0].imshow(best_chromosomes['color'])
        axes[1, 0].set_title('Evolved Color')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(best_chromosomes['grayscale'], cmap='gray', vmin=0, vmax=255)
        axes[1, 1].set_title('Evolved Grayscale')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(best_chromosomes['binary'], cmap='gray', vmin=0, vmax=255)
        axes[1, 2].set_title('Evolved Binary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        self.plot_fitness_comparison()
        
        for image_type in image_types:
            if image_type == 'color':
                target = self.color_target
            elif image_type == 'grayscale':
                target = self.grayscale_target
            else:
                target = self.binary_target
            
            final_fitness = self.calculate_fitness(best_chromosomes[image_type], target)
            max_fitness = 255 ** 2
            similarity_percentage = (final_fitness / max_fitness) * 100
            print(f"{image_type.capitalize()}: {similarity_percentage:.2f}% similarity")

if __name__ == "__main__":
    ga_evolution = GeneticImageEvolution(
        image_path="D:/Codings/Python/image generator_genetics/OIP.jpg",
        image_size=(100, 100), 
        population_size=100
    )
    ga_evolution.run_complete_evolution()