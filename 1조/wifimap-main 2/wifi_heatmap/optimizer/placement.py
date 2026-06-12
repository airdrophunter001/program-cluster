"""
유전 알고리즘 기반 AP 최적 배치 최적화
목적 함수: 커버리지 최대화 + SINR 최대화 + AP 수 최소화
"""

import random
import math
from dataclasses import dataclass, field
from typing import Callable, Optional
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from wifi_heatmap.core.propagation import PropagationParams
from wifi_heatmap.core.environment import BuildingEnvironment
from wifi_heatmap.core.router import Router, BAND_DEFAULTS
from wifi_heatmap.models.path_loss_model import PathLossModel


@dataclass
class Individual:
    routers: list[Router]
    fitness: float = -float("inf")
    coverage_pct: float = 0.0
    mean_rssi: float = -100.0
    mean_sinr: float = 0.0


@dataclass
class GAConfig:
    population_size: int = 30
    generations: int = 50
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    elite_size: int = 3
    tournament_size: int = 4
    target_coverage_pct: float = 90.0
    min_signal_dbm: float = -75.0
    w_coverage: float = 0.6
    w_sinr: float = 0.3
    w_penalty: float = 0.1


def _random_router(
    env: BuildingEnvironment,
    band: str,
    channel_pool: Optional[list[int]] = None,
) -> Router:
    x = random.uniform(0.5, env.width - 0.5)
    y = random.uniform(0.5, env.height - 0.5)
    channels = channel_pool or ([1, 6, 11] if band == "2.4GHz" else list(range(36, 165, 4)))
    ch = random.choice(channels)
    tx = random.uniform(15, 23)
    return Router(x=x, y=y, band=band, tx_power_dbm=tx, channel=ch)


def _evaluate(
    individual: Individual,
    model: PathLossModel,
    xs: np.ndarray,
    ys: np.ndarray,
    ga_cfg: GAConfig,
    use_walls: bool = True,
) -> Individual:
    best_rssi, sinr_grid, _ = model.compute_combined_map(
        individual.routers, xs, ys, use_walls
    )
    coverage = (best_rssi >= ga_cfg.min_signal_dbm).sum() / best_rssi.size * 100
    mean_rssi = float(np.mean(best_rssi[best_rssi > -100]))
    mean_sinr = float(np.clip(np.mean(sinr_grid), 0, 50))

    coverage_score = coverage / 100.0
    sinr_score = min(mean_sinr / 30.0, 1.0)
    penalty = max(0, (ga_cfg.target_coverage_pct - coverage) / 100.0)

    fitness = (
        ga_cfg.w_coverage * coverage_score
        + ga_cfg.w_sinr * sinr_score
        - ga_cfg.w_penalty * penalty
    )

    individual.fitness = fitness
    individual.coverage_pct = coverage
    individual.mean_rssi = mean_rssi
    individual.mean_sinr = mean_sinr
    return individual


def _tournament_select(population: list[Individual], k: int) -> Individual:
    contestants = random.sample(population, min(k, len(population)))
    return max(contestants, key=lambda ind: ind.fitness)


def _crossover(parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
    """포인트 교차: 공유기 위치 유전자 교환"""
    r1 = [deepcopy(r) for r in parent1.routers]
    r2 = [deepcopy(r) for r in parent2.routers]
    n = min(len(r1), len(r2))
    if n < 2:
        return Individual(routers=r1), Individual(routers=r2)
    pt = random.randint(1, n - 1)
    child1 = r1[:pt] + r2[pt:]
    child2 = r2[:pt] + r1[pt:]
    return Individual(routers=child1), Individual(routers=child2)


def _mutate(
    individual: Individual,
    env: BuildingEnvironment,
    mutation_rate: float,
    position_sigma: float = 1.5,
) -> Individual:
    """위치·채널·전력 돌연변이"""
    new_routers = []
    for router in individual.routers:
        r = deepcopy(router)
        if random.random() < mutation_rate:
            # 위치 돌연변이
            r.x = float(np.clip(r.x + np.random.normal(0, position_sigma), 0.5, env.width - 0.5))
            r.y = float(np.clip(r.y + np.random.normal(0, position_sigma), 0.5, env.height - 0.5))
        if random.random() < mutation_rate:
            # 채널 돌연변이
            channels = [1, 6, 11] if r.band == "2.4GHz" else list(range(36, 165, 4))
            r.channel = random.choice(channels)
        if random.random() < mutation_rate:
            # 전력 돌연변이
            r.tx_power_dbm = float(np.clip(r.tx_power_dbm + np.random.normal(0, 2), 10, 23))
        new_routers.append(r)
    return Individual(routers=new_routers)


@dataclass
class OptimizationResult:
    best_individual: Individual
    history: list[dict] = field(default_factory=list)
    converged: bool = False
    final_generation: int = 0

    @property
    def best_routers(self) -> list[Router]:
        return self.best_individual.routers

    @property
    def best_coverage_pct(self) -> float:
        return self.best_individual.coverage_pct

    def summary(self) -> dict:
        return {
            "coverage_pct": self.best_coverage_pct,
            "mean_rssi_dbm": self.best_individual.mean_rssi,
            "mean_sinr_db": self.best_individual.mean_sinr,
            "fitness": self.best_individual.fitness,
            "generations": self.final_generation,
            "converged": self.converged,
            "num_aps": len(self.best_individual.routers),
        }


class GeneticOptimizer:
    """유전 알고리즘 기반 AP 최적 배치"""

    def __init__(
        self,
        environment: BuildingEnvironment,
        propagation_params: PropagationParams,
        ga_config: GAConfig,
        band: str = "2.4GHz",
        noise_floor_dbm: float = -100.0,
    ):
        self.env = environment
        self.params = propagation_params
        self.ga_cfg = ga_config
        self.band = band
        self.model = PathLossModel(environment, propagation_params, noise_floor_dbm)

    def _init_population(
        self,
        num_aps: int,
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> list[Individual]:
        pop = []
        for _ in range(self.ga_cfg.population_size):
            routers = [_random_router(self.env, self.band) for _ in range(num_aps)]
            ind = Individual(routers=routers)
            _evaluate(ind, self.model, xs, ys, self.ga_cfg)
            pop.append(ind)
        return pop

    def optimize(
        self,
        num_aps: int,
        xs: np.ndarray,
        ys: np.ndarray,
        verbose: bool = True,
        use_walls: bool = True,
        callback: Optional[Callable[[int, float, float], None]] = None,
    ) -> OptimizationResult:
        population = self._init_population(num_aps, xs, ys)
        population.sort(key=lambda x: x.fitness, reverse=True)
        history: list[dict] = []
        best_overall = deepcopy(population[0])
        converged = False

        gen_iter = range(self.ga_cfg.generations)
        if verbose:
            gen_iter = tqdm(gen_iter, desc="Optimizing AP placement")

        no_improve_count = 0
        prev_best = -float("inf")

        for gen in gen_iter:
            # 엘리트 보존
            new_pop = [deepcopy(ind) for ind in population[:self.ga_cfg.elite_size]]

            while len(new_pop) < self.ga_cfg.population_size:
                if random.random() < self.ga_cfg.crossover_rate and len(population) >= 2:
                    p1 = _tournament_select(population, self.ga_cfg.tournament_size)
                    p2 = _tournament_select(population, self.ga_cfg.tournament_size)
                    c1, c2 = _crossover(p1, p2)
                    for child in [c1, c2]:
                        child = _mutate(child, self.env, self.ga_cfg.mutation_rate)
                        _evaluate(child, self.model, xs, ys, self.ga_cfg, use_walls)
                        new_pop.append(child)
                else:
                    ind = _tournament_select(population, self.ga_cfg.tournament_size)
                    ind = _mutate(deepcopy(ind), self.env, self.ga_cfg.mutation_rate)
                    _evaluate(ind, self.model, xs, ys, self.ga_cfg, use_walls)
                    new_pop.append(ind)

            population = sorted(new_pop[:self.ga_cfg.population_size],
                                 key=lambda x: x.fitness, reverse=True)

            gen_best = population[0]
            if gen_best.fitness > best_overall.fitness:
                best_overall = deepcopy(gen_best)

            history.append({
                "generation": gen,
                "best_fitness": gen_best.fitness,
                "best_coverage_pct": gen_best.coverage_pct,
                "mean_fitness": np.mean([ind.fitness for ind in population]),
                "best_sinr": gen_best.mean_sinr,
            })

            if verbose:
                gen_iter.set_postfix({
                    "cov": f"{gen_best.coverage_pct:.1f}%",
                    "fit": f"{gen_best.fitness:.4f}",
                })

            if callback:
                callback(gen, gen_best.coverage_pct, gen_best.fitness)

            # 수렴 판정
            if abs(gen_best.fitness - prev_best) < 1e-5:
                no_improve_count += 1
            else:
                no_improve_count = 0
            prev_best = gen_best.fitness

            if no_improve_count >= 10:
                converged = True
                if verbose:
                    print(f"\nConverged at generation {gen}")
                break

            if gen_best.coverage_pct >= self.ga_cfg.target_coverage_pct:
                converged = True
                if verbose:
                    print(f"\nTarget coverage reached at generation {gen}")
                break

        return OptimizationResult(
            best_individual=best_overall,
            history=history,
            converged=converged,
            final_generation=len(history),
        )


def plot_optimization_history(result: OptimizationResult) -> "plt.Figure":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = result.history
    gens = [h["generation"] for h in history]
    best_cov = [h["best_coverage_pct"] for h in history]
    best_fit = [h["best_fitness"] for h in history]
    mean_fit = [h["mean_fitness"] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(gens, best_cov, "g-o", markersize=4, linewidth=2, label="Best Coverage")
    ax1.axhline(y=result.best_coverage_pct, color="green", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Coverage (%)")
    ax1.set_ylim(0, 105)
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax1.set_title("Genetic Algorithm Optimization History")

    ax2.plot(gens, best_fit, "b-o", markersize=4, linewidth=2, label="Best Fitness")
    ax2.plot(gens, mean_fit, "r--", linewidth=1.5, alpha=0.7, label="Mean Fitness")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Fitness")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    return fig
