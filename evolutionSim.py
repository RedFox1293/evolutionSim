"""
Evolution Simulator
===================
A real-time evolution simulation with heritable traits, natural selection,
mutation, speciation, and environmental adaptation.

Requirements:
    pip install pygame-ce numpy pandas matplotlib

Controls (Pygame window):
    SPACE  — Pause / Resume
    F      — Cycle speed  (1x -> 5x -> 20x -> 100x)
    Q      — Quit and generate analysis
    Scroll — Scroll HUD species list

Run:
    python evolutionSim.py
"""

import math
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import pygame  # type: ignore  # installed as pygame-ce: pip install pygame-ce
except ImportError:
    print("ERROR: pygame is not installed.")
    print("Install with:  pip install pygame-ce")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    # Grid
    "grid_width": 80,
    "grid_height": 60,
    # Population
    "starting_species": 8,
    "pop_cap": 5000,            # global hard ceiling (emergency OOM guard only)
    "prey_cap": 400,            # per-species K for a pure prey species  (surv→0)
    "pred_cap": 40,             # per-species K for a pure predator species (surv→1)
    "initial_pop_per_species": 20,   # fallback if initial_pops not set
    # Per-species starting counts (index matches INITIAL_SPECIES_TRAITS).
    # Two clusters of competing predator species (will fight to dominate their niche),
    # and four competing prey species (most will eventually be displaced).
    "initial_pops": [22, 20, 14, 16, 22, 26, 30, 34],
    # Environment
    "env_drift_freq": 300,          # steps between environmental drift events
    # Mutation
    "mutation_bias": 0.60,          # fraction of mutations that are adverse
    # Speciation
    "speciation_threshold": 0.30,   # normalised trait-space distance threshold (lower = easier fork)
    "speciation_min_outliers": 4,   # min persistent outliers to trigger fork
    "speciation_outlier_steps": 4,  # consecutive steps individual must be outlier
    "jump_speciation_threshold": 0.50,  # large mutation → accumulate outlier_steps 3x faster
    # Macro-mutations: rare large jumps that rapidly diverge a lineage
    "macro_mut_prob":  0.008,       # per-offspring probability of a macro-mutation event
    "macro_mut_scale": 4.0,         # all trait deltas scaled up by this factor during macro events
    # History
    "history_interval": 10,         # steps between history snapshots
    # Coupling constants  rep_rate ~ k / survivability^alpha
    "k_coupling": 0.08,
    "alpha_coupling": 1.50,
    "max_rep_rate": 3.0,
    # Predation (Lotka-Volterra -b·P·Q term)
    "predation_beta": 2.5,      # neighbourhood predation coefficient
    # Kill-bonus replication: expected offspring per kill = surv * kill_rep_scale (Poisson)
    "kill_rep_scale": 0.80,     # predators must rely on kills; apex pred gets ~0.7 offspring/kill
    # Trophic replication suppression: base_rep *= (1-surv)^trophic_exp
    # so high-surv predators barely self-replicate and truly depend on kill bonuses.
    "trophic_exp": 1.5,
    # Battle
    "battle_sharpness": 0.85,       # amplifies win-prob spread; higher = more decisive battles
    # Pygame window
    "window_width": 1200,
    "window_height": 800,
    "hud_height": 180,
    "topbar_height": 30,
}

# ── Module-level constants ────────────────────────────────────────────────────
TRAIT_NAMES = ["rep_age", "survivability", "rep_rate", "mutation_rate", "env_score"]

# Adverse mutation direction: -1 = decrease is bad, +1 = increase is bad
TRAIT_ADVERSE_DIR = {
    "rep_age":       -1,
    "survivability": -1,
    "rep_rate":      -1,
    "mutation_rate": +1,
    "env_score":     -1,
}

# Hard clamp bounds after mutation
TRAIT_RANGES = {
    "rep_age":       (5.0,   200.0),
    "survivability": (0.01,  1.0),
    "rep_rate":      (0.05,  3.0),
    "mutation_rate": (0.001, 1.0),
    "env_score":     (0.0,   1.0),
}

# Per-trait Gaussian sigma = effective_mutation_rate * TRAIT_SCALES[trait]
TRAIT_SCALES = {
    "rep_age":       15.0,
    "survivability":  0.15,
    "env_score":      0.15,
    "mutation_rate":  0.10,
}

# 12 visually distinct RGB colours assigned to species
COLOR_PALETTE = [
    (220,  50,  47),   # red
    ( 38, 139, 210),   # blue
    (133, 153,   0),   # olive-green
    (211,  54, 130),   # magenta
    ( 42, 161, 152),   # teal
    (253, 200,  50),   # gold
    (108, 113, 196),   # violet
    (203,  75,  22),   # orange-red
    (  0, 200, 100),   # bright-green
    (255, 120,  50),   # orange
    (160,  80, 200),   # purple
    (100, 200, 230),   # sky-blue
]

SHAPES = ["circle", "triangle", "square", "diamond", "pentagon", "hexagon"]

# Hard-coded initial 8 species — seeded as three ecological clusters that will
# compete internally, with natural selection and speciation producing the trophic
# pyramid over time.  rep_rate is computed from k/s^alpha at initialisation.
#
#   Cluster A — two high-surv predators that will compete to dominate that niche:
#     Index 0: apex-contender  (surv=0.88, very slow, very stable)
#     Index 1: rival predator  (surv=0.74, faster but sloppier)
#
#   Cluster B — two mid-surv mesopredators / generalists:
#     Index 2: alpha generalist (surv=0.52)
#     Index 3: beta generalist  (surv=0.38)
#
#   Cluster C — four prey/fast-breeder species competing in the bottom niche:
#     Index 4: surv=0.22  (slow prey — will likely be outcompeted)
#     Index 5: surv=0.12
#     Index 6: surv=0.06
#     Index 7: surv=0.02  (abundant, ultra-fast breeders)
INITIAL_SPECIES_TRAITS = [
    {"survivability": 0.88, "rep_age": 108.0, "mutation_rate": 0.03, "env_score": 0.78},
    {"survivability": 0.74, "rep_age":  85.0, "mutation_rate": 0.08, "env_score": 0.70},
    {"survivability": 0.52, "rep_age":  60.0, "mutation_rate": 0.16, "env_score": 0.60},
    {"survivability": 0.38, "rep_age":  45.0, "mutation_rate": 0.26, "env_score": 0.50},
    {"survivability": 0.22, "rep_age":  30.0, "mutation_rate": 0.36, "env_score": 0.40},
    {"survivability": 0.12, "rep_age":  18.0, "mutation_rate": 0.46, "env_score": 0.32},
    {"survivability": 0.06, "rep_age":  11.0, "mutation_rate": 0.56, "env_score": 0.25},
    {"survivability": 0.02, "rep_age":   7.0, "mutation_rate": 0.65, "env_score": 0.18},
]


# ── Utility functions ─────────────────────────────────────────────────────────
def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def coupled_rep_rate(survivability: float, k: float = 0.30,
                     alpha: float = 1.50, max_rr: float = 8.0) -> float:
    """Compute the theoretically-coupled replication rate from survivability."""
    return clamp(k / max(survivability, 0.001) ** alpha, 0.05, max_rr)


# ── Individual ────────────────────────────────────────────────────────────────
class Individual:
    """
    A single organism living on the 2-D toroidal grid.

    Each individual carries its own trait dict, which may diverge from the
    parent species mean through successive mutations.  Life-history parameters
    (personal replication stopping age, maximum lifespan) are drawn once at
    birth and remain fixed.
    """

    _id_counter: int = 0

    def __init__(self, species_id: int, traits: dict,
                 position: tuple, step_born: int) -> None:
        Individual._id_counter += 1
        self.id: int = Individual._id_counter
        self.species_id: int = species_id
        self.traits: dict = dict(traits)
        self.x, self.y = position
        self.age: int = 0
        self.step_born: int = step_born

        # Personal replication stopping age ~ N(rep_age, rep_age * 0.1)
        ra = traits["rep_age"]
        self.personal_rep_age: float = max(1.0, random.gauss(ra, ra * 0.1))

        # Individual max lifespan ~ N(survivability * 150 + 50, 20), clamped [10, 500]
        surv = traits["survivability"]
        self.max_lifespan: int = int(
            clamp(round(random.gauss(surv * 150 + 50, 20)), 10, 500)
        )

        # Consecutive steps this individual has been a speciation outlier
        self.outlier_steps: int = 0

        # Accumulated genomic instability from sustained high mutation rate.
        # Ranges 0–1; penalises rep_rate in _process_replication.
        self.mutation_burden: float = 0.0

    def compute_fitness_bonus(self, env_optimum: float) -> float:
        """Multiplicative fitness bonus based on environmental alignment."""
        return 1.0 + 0.3 * (1.0 - abs(self.traits["env_score"] - env_optimum))

    def effective_survivability(self, env_optimum: float) -> float:
        """Survivability scaled by the current environmental fitness bonus.

        A genetic-stability bonus rewards low mutation rates: individuals that
        mutate slowly are more genetically coherent and up to 15% tougher.
        """
        base = self.traits["survivability"] * self.compute_fitness_bonus(env_optimum)
        stability_bonus = 1.0 + 0.15 * (1.0 - self.traits["mutation_rate"])
        return base * stability_bonus

    def mutate(self, env_optimum: float, macro: bool = False) -> dict:
        """
        Produce a mutated trait dictionary for an offspring.

        Normal mode: per-trait Gaussian deltas scaled by eff_mut.
        Macro mode (macro=True): all deltas are scaled up by CONFIG["macro_mut_scale"],
        modelling rare large-jump mutations (punctuated equilibrium events). These
        offspring diverge rapidly from the parent species mean, accelerating speciation.

        The survivability <-> rep_rate inverse coupling is maintained at
        70% weight toward the theoretical value with 30% free drift.
        """
        t = self.traits
        # Effective mutation rate reduced when env_score is high (well-adapted)
        eff_mut = t["mutation_rate"] * (1.0 - 0.5 * t["env_score"])
        # Macro-mutation amplifier: large rare events jump far in trait space.
        ms = CONFIG.get("macro_mut_scale", 4.0) if macro else 1.0
        new = dict(t)

        # rep_age: 75% adverse (decreases) — lifespans erode unless selected against.
        ra_sign = -1 if random.random() < 0.75 else 1
        new["rep_age"] = clamp(
            t["rep_age"] + abs(random.gauss(0.0, eff_mut * TRAIT_SCALES["rep_age"] * ms)) * ra_sign,
            *TRAIT_RANGES["rep_age"]
        )

        # survivability: NEUTRAL mutation (50 / 50) — both predator and prey are
        # viable evolutionary strategies, so there is no single "adverse" direction.
        # Battle selection and trophic kill-bonuses alone determine which surv level
        # is stable for a given lineage.
        surv_sign = 1 if random.random() < 0.50 else -1
        new["survivability"] = clamp(
            t["survivability"] + abs(random.gauss(0.0, eff_mut * TRAIT_SCALES["survivability"] * ms)) * surv_sign,
            *TRAIT_RANGES["survivability"]
        )

        # env_score: 90% of mutations move TOWARD the environmental optimum,
        # modelling adaptive pressure: well-adapted offspring are more viable.
        env_toward_optimum = 1.0 if t["env_score"] < env_optimum else -1.0
        env_sign = (env_toward_optimum if random.random() < CONFIG["mutation_bias"]
                    else -env_toward_optimum)
        env_delta = abs(random.gauss(0.0, eff_mut * TRAIT_SCALES["env_score"] * ms)) * env_sign
        new["env_score"] = clamp(t["env_score"] + env_delta, *TRAIT_RANGES["env_score"])

        # mutation_rate mutates at rate * 0.1  (adverse = increase)
        mr_sign = (TRAIT_ADVERSE_DIR["mutation_rate"]
                   if random.random() < CONFIG["mutation_bias"]
                   else -TRAIT_ADVERSE_DIR["mutation_rate"])
        mr_delta = abs(random.gauss(0.0, t["mutation_rate"] * 0.1 * ms)) * mr_sign
        new["mutation_rate"] = clamp(
            t["mutation_rate"] + mr_delta, *TRAIT_RANGES["mutation_rate"]
        )

        # rep_rate: 70% pull toward theoretical coupling, 30% free drift
        theoretical = coupled_rep_rate(
            new["survivability"],
            CONFIG["k_coupling"], CONFIG["alpha_coupling"], CONFIG["max_rep_rate"]
        )
        free_drift = clamp(
            t["rep_rate"] + random.gauss(0.0, eff_mut * 0.5 * ms),
            *TRAIT_RANGES["rep_rate"]
        )
        new["rep_rate"] = clamp(
            0.70 * theoretical + 0.30 * free_drift, *TRAIT_RANGES["rep_rate"]
        )
        return new

    def death_probability(self) -> float:
        """
        Per-step probability of natural death.

        Stays at a low base until the individual passes 60% of its max lifespan,
        then ramps linearly to 0.05 at max_lifespan.
        """
        base = 0.002
        threshold = self.max_lifespan * 0.6
        if self.age <= threshold:
            return base
        ramp = min(1.0, (self.age - threshold) / max(1.0, self.max_lifespan - threshold))
        return base + ramp * (0.05 - base)

    def step_age(self) -> None:
        """Increment age by one simulation step and update mutation burden."""
        self.age += 1
        # Genomic instability builds when mutation_rate stays above threshold;
        # decays when the individual stabilises.  At burden=1.0 the individual's
        # effective rep_rate is reduced to 15% of its base value.
        _MUT_HIGH = 0.25
        if self.traits["mutation_rate"] > _MUT_HIGH:
            self.mutation_burden = min(
                1.0, self.mutation_burden + 0.05 * self.traits["mutation_rate"]
            )
        else:
            self.mutation_burden = max(0.0, self.mutation_burden - 0.03)

    @classmethod
    def reset_counter(cls) -> None:
        cls._id_counter = 0


# ── Species ───────────────────────────────────────────────────────────────────
class Species:
    """
    Metadata record for a species lineage.

    Tracks mean traits (updated each step from living members), colour, shape,
    ancestry, and peak population.  Individual members may deviate from these
    means due to mutation.
    """

    _shape_index: int = 0

    def __init__(self, species_id: int, parent_id,
                 mean_traits: dict, color: tuple, step_born: int) -> None:
        self.id: int = species_id
        self.parent_id = parent_id          # int | None
        self.mean_traits: dict = dict(mean_traits)
        self.color: tuple = color
        self.shape: str = SHAPES[Species._shape_index % len(SHAPES)]
        Species._shape_index += 1
        self.step_born: int = step_born
        self.extinct_step = None            # int | None
        self.alive: bool = True
        self.peak_population: int = 0
        self.speciation_events_spawned: int = 0

    def update_mean_traits(self, members: list) -> None:
        """Recompute mean traits from a list of living Individual objects."""
        if not members:
            return
        n = len(members)
        for trait in TRAIT_NAMES:
            self.mean_traits[trait] = sum(m.traits[trait] for m in members) / n
        if n > self.peak_population:
            self.peak_population = n

    def mark_extinct(self, step: int) -> None:
        """Record that this species went extinct at the given step."""
        self.alive = False
        self.extinct_step = step

    def trait_distance(self, individual: "Individual") -> float:
        """
        Normalised Euclidean distance in 5-D trait space between this species'
        mean and an individual's traits.  Result is roughly in [0, 1].
        """
        sq = 0.0
        for trait in TRAIT_NAMES:
            lo, hi = TRAIT_RANGES[trait]
            span = hi - lo
            ind_n  = (individual.traits[trait] - lo) / span
            mean_n = (self.mean_traits[trait]  - lo) / span
            sq += (ind_n - mean_n) ** 2
        return math.sqrt(sq / len(TRAIT_NAMES))

    def lifespan_steps(self, current_step: int) -> int:
        """Steps this species has been (or was) alive."""
        end = self.extinct_step if self.extinct_step is not None else current_step
        return end - self.step_born

    @classmethod
    def reset_shape_index(cls) -> None:
        cls._shape_index = 0


# ── SimulationEngine ──────────────────────────────────────────────────────────
class SimulationEngine:
    """
    Core simulation engine.

    Manages the 2-D toroidal grid, processes movement, encounters (battles),
    replication, natural death, speciation, and environmental drift.
    Records a step-wise history and event log for post-simulation analysis.

    The spatial grid is stored as a dict: (x, y) -> [individual_id, ...].
    Cells absent from the dict are empty.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.step_count: int = 0
        self.running: bool = True
        self.env_optimum: float = 0.8

        # Spatial index
        self.grid: dict = {}
        # Primary stores
        self.individuals: dict = {}
        self.species: dict = {}
        self._species_id_counter: int = 0

        # Deferred replication from battle wins
        self._replication_queue: list = []

        # History and event log
        self.history: list = []
        self.event_log: list = []  # NOTE: can grow large over long runs

        # Colour reuse suppression: color -> step of extinction
        self._recently_extinct_colors: dict = {}

        Individual.reset_counter()
        Species.reset_shape_index()
        self._initialize_population()

    # ── Initialisation ───────────────────────────────────────────────────────
    def _initialize_population(self) -> None:
        """Create the 5 initial species with distinct hand-crafted trait profiles."""
        palette = list(COLOR_PALETTE)
        random.shuffle(palette)
        k      = self.config["k_coupling"]
        alpha  = self.config["alpha_coupling"]
        max_rr = self.config["max_rep_rate"]
        # Per-species starting counts (supports trophic pyramid: few predators, many prey)
        default_n = self.config.get("initial_pop_per_species", 20)
        initial_pops = self.config.get("initial_pops", [])
        for i, base in enumerate(INITIAL_SPECIES_TRAITS):
            traits = dict(base)
            traits["rep_rate"] = coupled_rep_rate(traits["survivability"], k, alpha, max_rr)
            sp = self._create_species(parent_id=None, mean_traits=traits,
                                      color=palette[i % len(palette)])
            n = initial_pops[i] if i < len(initial_pops) else default_n
            for _ in range(n):
                x = random.randint(0, self.config["grid_width"]  - 1)
                y = random.randint(0, self.config["grid_height"] - 1)
                self._add_individual(Individual(sp.id, traits, (x, y), self.step_count))

    def _create_species(self, parent_id, mean_traits: dict, color: tuple) -> Species:
        """Allocate and register a new Species object."""
        self._species_id_counter += 1
        sp = Species(self._species_id_counter, parent_id,
                     mean_traits, color, self.step_count)
        self.species[sp.id] = sp
        return sp

    # ── Main step ────────────────────────────────────────────────────────────
    def step(self) -> None:
        """Advance the simulation by one time step."""
        if not self.individuals:
            return

        self._age_individuals()
        self._move_individuals()
        self._process_encounters()     # one battle per occupied cell → fills queue
        self._process_predation()      # LV neighbourhood sweep → fills queue
        self._process_replication()    # survivors reproduce, drains queue
        self._process_natural_death()

        if self.step_count > 0 and self.step_count % self.config["env_drift_freq"] == 0:
            self._update_environment()

        self._update_outlier_counters()
        self._update_species_means()

        if self.step_count % 25 == 0:
            self._check_speciation_forks()

        self._check_extinctions()

        if self.step_count % self.config["history_interval"] == 0:
            self._record_history()

        self.step_count += 1

    # ── Sub-steps ────────────────────────────────────────────────────────────
    def _age_individuals(self) -> None:
        for ind in self.individuals.values():
            ind.step_age()

    def _move_individuals(self) -> None:
        W = self.config["grid_width"]
        H = self.config["grid_height"]
        for ind in list(self.individuals.values()):
            old = (ind.x, ind.y)
            old_list = self.grid.get(old, [])
            if ind.id in old_list:
                old_list.remove(ind.id)
            if not old_list and old in self.grid:
                del self.grid[old]

            ind.x = (ind.x + random.randint(1, 3) * random.choice((-1, 1))) % W
            ind.y = (ind.y + random.randint(1, 3) * random.choice((-1, 1))) % H
            self.grid.setdefault((ind.x, ind.y), []).append(ind.id)

    def _process_encounters(self) -> None:
        """Resolve one battle per occupied cell that has >= 2 individuals."""
        to_remove: list = []
        for id_list in list(self.grid.values()):
            # Filter to IDs still alive this step
            valid = [i for i in id_list
                     if i in self.individuals and i not in to_remove]
            if len(valid) < 2:
                continue

            # Prefer same-tier battles: pick id_a randomly, then bias id_b toward
            # a similar survivability opponent.  Weight = 1/(0.05 + |surv_a − surv_b|).
            # This keeps apex predators fighting each other (not just wiping out mid-tier)
            # and lets prey compete among themselves; cross-tier predation happens via
            # the neighbourhood predation sweep (_process_predation).
            id_a = random.choice(valid)
            ind_a = self.individuals[id_a]
            others = [i for i in valid if i != id_a]
            if len(others) == 1:
                id_b = others[0]
            else:
                s_a = ind_a.traits["survivability"]
                wts = [1.0 / (0.05 + abs(s_a - self.individuals[i].traits["survivability"]))
                       for i in others]
                total_w = sum(wts)
                r = random.random() * total_w
                cum = 0.0
                id_b = others[-1]
                for i, w in zip(others, wts):
                    cum += w
                    if r < cum:
                        id_b = i
                        break
            ind_b = self.individuals[id_b]

            eff_a = ind_a.effective_survivability(self.env_optimum)
            eff_b = ind_b.effective_survivability(self.env_optimum)
            total = max(eff_a + eff_b, 1e-9)

            # Amplified win-probability: sharper gap between high/low survivability.
            # Formula: 0.5 + sharpness * (eff_a - eff_b) / total, clamped [0.05, 0.95].
            # At sharpness=0.5 this equals the original eff_a/total formula.
            sharpness = self.config.get("battle_sharpness", 0.75)
            p_a_wins  = clamp(0.5 + sharpness * (eff_a - eff_b) / total, 0.05, 0.95)

            if random.random() < p_a_wins:
                winner, loser = ind_a, ind_b
            else:
                winner, loser = ind_b, ind_a

            self._log_event("battle_death", loser)
            to_remove.append(loser.id)

            # Kill-replication bonus: Poisson-distributed offspring from each kill.
            # Expected offspring = surv * kill_rep_scale, giving apex predators
            # ~1.4 expected bonus offspring per battle win.  Low-surv prey gain
            # almost nothing (expected ~0.075 per kill — they rely on base rep_rate).
            kill_scale = self.config.get("kill_rep_scale", 1.5)
            kill_expected = winner.traits["survivability"] * kill_scale
            n_bonus = min(int(np.random.poisson(kill_expected)), 5)
            for _ in range(n_bonus):
                self._replication_queue.append(winner.id)

        for ind_id in to_remove:
            self._remove_individual(ind_id)

    def _species_cap(self, sp: "Species") -> int:
        """
        Per-species carrying capacity K.

        Below threshold (surv ≤ 0.55): independent per-species K.
            surv=0.02 → K ≈ 393   surv=0.52 → K ≈ 213
        Mid-tier generalists (0.35 < surv ≤ 0.55) get a moderate independent K
        that lets them coexist without competing against apex predators.

        True apex predators (surv > 0.55): share a fixed ecological budget
        (pred_k × 5 = 200 total).  The more apex species coexist, the smaller
        each one's K, so they compete directly — one winner gradually claims
        the full budget as rivals go extinct.
            1 apex species → K = 200
            2 apex species → K = 100 each
            4 apex species → K = 50 each
        """
        surv = sp.mean_traits.get("survivability", 0.5)
        prey_k = self.config.get("prey_cap", 400)
        pred_k = self.config.get("pred_cap", 40)
        pred_threshold = 0.55

        if surv > pred_threshold:
            # Count living predator species (shared budget)
            n_pred = max(1, sum(
                1 for s in self.species.values()
                if s.alive and s.mean_traits.get("survivability", 0) > pred_threshold
            ))
            total_pred_budget = pred_k * 5   # = 200 total ecological space for predators
            return max(5, round(total_pred_budget / n_pred))

        # Prey: per-species K scales with preyness
        return max(5, round(prey_k * (1.0 - surv) + pred_k * surv))

    def _process_replication(self) -> None:
        """
        Produce offspring via Lotka-Volterra per-species logistic growth.

        Each species has its own carrying capacity K derived from its mean
        survivability (high-surv predators → small K; low-surv prey → large K).
        The logistic term is linear: max(0, 1 - sp_pop / K_sp), which keeps
        replication proportional to remaining niche space and lets predation
        meaningfully suppress prey below K.

        Kill-bonus offspring from _replication_queue (battles + predation) are
        also resolved here.  Predators may briefly overshoot their K by up to
        2× during a prey boom — consistent with LV oscillatory dynamics.

        A global hard cap (pop_cap) guards against memory overflow only.
        """
        hard_cap  = self.config["pop_cap"]
        total_pop = len(self.individuals)

        # Per-species effective population (updated live as offspring are added,
        # so later individuals in the same step see the updated logistic).
        species_pops: dict = defaultdict(int)
        for ind in self.individuals.values():
            species_pops[ind.species_id] += 1

        new_offspring: list = []

        for ind in list(self.individuals.values()):
            if total_pop + len(new_offspring) >= hard_cap:
                break
            if ind.age >= ind.personal_rep_age:
                continue
            sp = self.species.get(ind.species_id)
            if sp is None:
                continue
            K_sp     = self._species_cap(sp)
            # Linear logistic uses live count so no within-step overshoot occurs
            sp_pop   = species_pops[ind.species_id]
            logistic = max(0.0, 1.0 - sp_pop / K_sp)
            if logistic <= 0.0:
                continue
            # Mutation burden: sustained high mutation_rate corrodes rep_rate
            burden_penalty = max(0.15, 1.0 - ind.mutation_burden)
            # Environmental fitness bonus: well-adapted individuals breed faster
            env_boost = ind.compute_fitness_bonus(self.env_optimum)
            # Trophic suppression: high-survivability predators self-replicate very
            # little — they must depend on kill bonuses to grow their population.
            # (1-surv)^exp → apex (surv=0.88): 0.0028; prey (surv=0.02): 0.94
            trophic_exp = self.config.get("trophic_exp", 1.5)
            trophic_factor = max(0.005, (1.0 - ind.traits["survivability"]) ** trophic_exp)
            lam = ind.traits["rep_rate"] * logistic * burden_penalty * env_boost * trophic_factor
            if lam <= 0.0:
                continue
            macro_prob = self.config.get("macro_mut_prob", 0.0)
            n = min(int(np.random.poisson(lam)), 10)
            for _ in range(n):
                if total_pop + len(new_offspring) >= hard_cap:
                    break
                if species_pops[ind.species_id] >= K_sp:
                    break                                   # hit K mid-loop
                macro = random.random() < macro_prob
                child = Individual(ind.species_id,
                                   ind.mutate(self.env_optimum, macro=macro),
                                   (ind.x, ind.y), self.step_count)
                new_offspring.append(child)
                species_pops[child.species_id] += 1         # live update

        # Kill-bonus replication (battle winners + predation kills).
        # This is the PRIMARY reproduction channel for predators (trophic_factor
        # nearly eliminates their self-replication).  Strictly capped at species K.
        for ind_id in self._replication_queue:
            if ind_id not in self.individuals:
                continue
            if total_pop + len(new_offspring) >= hard_cap:
                break
            ind = self.individuals[ind_id]
            if ind.age >= ind.personal_rep_age:
                continue
            sp = self.species.get(ind.species_id)
            if sp is None:
                continue
            sp_pop = species_pops[ind.species_id]
            K_sp   = self._species_cap(sp)
            if sp_pop >= K_sp:            # strict cap — no overshoot
                continue
            macro = random.random() < self.config.get("macro_mut_prob", 0.0)
            child = Individual(ind.species_id,
                               ind.mutate(self.env_optimum, macro=macro),
                               (ind.x, ind.y), self.step_count)
            new_offspring.append(child)
            species_pops[ind.species_id] += 1

        self._replication_queue.clear()

        for child in new_offspring:
            self._add_individual(child)
            self._log_event("birth", child)

    def _process_natural_death(self) -> None:
        """Apply stochastic age-dependent natural death to every individual."""
        to_remove = [
            ind_id for ind_id, ind in list(self.individuals.items())
            if random.random() < ind.death_probability()
        ]
        for ind_id in to_remove:
            if ind_id in self.individuals:
                self._log_event("natural_death", self.individuals[ind_id])
                self._remove_individual(ind_id)

    def _process_predation(self) -> None:
        """
        Lotka-Volterra neighbourhood predation sweep  (-b·P·Q term).

        Each individual contributes its survivability as predator weight to its
        own grid cell.  For every prey-like individual (preyness = 1−surv > 0.15)
        the local predator weight in the 3×3 Moore neighbourhood is summed; the
        per-step death probability is:

            p_die = min(beta × preyness × local_pred_weight, 0.5)

        Predators that cause a kill earn a bonus entry in _replication_queue
        (the +c·P·Q term), which _process_replication will drain next.
        """
        if not self.individuals:
            return

        beta = self.config.get("predation_beta", 2.0)
        W    = self.config["grid_width"]
        H    = self.config["grid_height"]

        # ── Build predator-weight map (cell → Σ survivability) ──────────────
        pred_weight: dict = defaultdict(float)
        hunters_in_cell: dict = defaultdict(list)
        for ind in self.individuals.values():
            surv = ind.traits["survivability"]
            cell = (ind.x, ind.y)
            pred_weight[cell] += surv
            if surv > 0.35:                   # meaningful predators only
                hunters_in_cell[cell].append(ind)

        # ── Pre-compute 3×3 neighbourhood sums ──────────────────────────────
        neighbourhood_pred: dict = {}
        for ind in self.individuals.values():
            cx, cy = ind.x, ind.y
            if (cx, cy) in neighbourhood_pred:
                continue
            neighbourhood_pred[(cx, cy)] = sum(
                pred_weight.get(((cx + dx) % W, (cy + dy) % H), 0.0)
                for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            )

        # ── Predation mortality pass ─────────────────────────────────────────
        to_remove: list = []
        for ind in list(self.individuals.values()):
            preyness = 1.0 - ind.traits["survivability"]
            if preyness < 0.15:               # near-pure predators aren't prey
                continue
            local_pred = neighbourhood_pred.get((ind.x, ind.y), 0.0)
            # Subtract own contribution — individuals can't eat themselves
            local_pred = max(0.0, local_pred - ind.traits["survivability"])
            if local_pred <= 0.0:
                continue
            p_die = min(beta * preyness * local_pred, 0.5)
            if random.random() < p_die:
                to_remove.append(ind.id)

        # ── Resolve deaths and award kill-replication bonuses ────────────────
        for ind_id in to_remove:
            if ind_id not in self.individuals:
                continue
            prey = self.individuals[ind_id]
            self._log_event("predation_death", prey)
            # Find the nearest hunter and give it a replication chance
            hunters = hunters_in_cell.get((prey.x, prey.y), [])
            if not hunters:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        hunters = hunters + hunters_in_cell.get(
                            ((prey.x + dx) % W, (prey.y + dy) % H), []
                        )
            if hunters:
                killer = random.choice(hunters)
                if killer.id in self.individuals:
                    # Predation kill bonus: 80% × surv probability per kill
                    if random.random() < killer.traits["survivability"] * 0.8:
                        self._replication_queue.append(killer.id)
            self._remove_individual(ind_id)

    def _update_environment(self) -> None:
        """Randomly drift the environmental optimum, shifting selective pressures.

        Uses a larger sigma (0.12) so each environmental event is a meaningful
        shock that differentially affects well- vs. poorly-adapted individuals,
        creating divergent selection pressure that drives speciation.
        """
        self.env_optimum = clamp(
            self.env_optimum + random.gauss(0.0, 0.12), 0.0, 1.0
        )

    def _update_outlier_counters(self) -> None:
        """
        Track how many consecutive steps each individual has been distant
        from its assigned species mean.  Used by speciation detection.
        """
        by_species: dict = defaultdict(list)
        for ind in self.individuals.values():
            by_species[ind.species_id].append(ind)

        threshold   = self.config["speciation_threshold"]
        jump_thresh = self.config.get("jump_speciation_threshold", 0.65)
        for sid, members in by_species.items():
            sp = self.species.get(sid)
            if sp is None or not sp.alive:
                continue
            for ind in members:
                dist = sp.trait_distance(ind)
                if dist > jump_thresh:
                    # Large mutation — accumulates 3× faster so forks sooner
                    ind.outlier_steps += 3
                elif dist > threshold:
                    ind.outlier_steps += 1
                else:
                    # Slow decay instead of hard reset: accumulated divergence
                    # is not wiped out by a single step back near the species mean.
                    ind.outlier_steps = max(0, ind.outlier_steps - 1)

    def _update_species_means(self) -> None:
        """Recompute each living species' mean traits from its current members."""
        by_species: dict = defaultdict(list)
        for ind in self.individuals.values():
            by_species[ind.species_id].append(ind)
        for sid, sp in self.species.items():
            if sp.alive:
                sp.update_mean_traits(by_species.get(sid, []))

    def _check_speciation_forks(self) -> None:
        """
        Fork a subpopulation into a new species when enough individuals have
        been persistent outliers in normalised trait-space.
        """
        min_outliers = self.config["speciation_min_outliers"]
        min_steps    = self.config["speciation_outlier_steps"]
        for sid in list(self.species.keys()):
            sp = self.species[sid]
            if not sp.alive:
                continue
            members = [ind for ind in self.individuals.values()
                       if ind.species_id == sid]
            if len(members) <= 10:
                continue
            outliers = [ind for ind in members
                        if ind.outlier_steps >= min_steps]
            if len(outliers) >= min_outliers:
                self._fork_species(sp, outliers)

    def _fork_species(self, parent: Species, outliers: list) -> None:
        """Create a new daughter species from divergent outlier individuals.

        Beyond transferring the outliers, we clone a handful of them into
        nearby positions so the daughter species starts with enough individuals
        to survive early competition and not immediately go extinct.
        """
        new_mean = {t: sum(ind.traits[t] for ind in outliers) / len(outliers)
                    for t in TRAIT_NAMES}
        new_sp = self._create_species(parent_id=parent.id,
                                       mean_traits=new_mean,
                                       color=self._get_color_for_new_species())
        parent.speciation_events_spawned += 1
        for ind in outliers:
            ind.species_id  = new_sp.id
            ind.outlier_steps = 0

        # Seed the daughter species with a few extra clones so it has a viable
        # founding population rather than surviving on only the outlier handful.
        W = self.config["grid_width"]
        H = self.config["grid_height"]
        n_seeds = min(8, len(outliers) * 2)
        for _ in range(n_seeds):
            source = random.choice(outliers)
            nx = (source.x + random.randint(-3, 3)) % W
            ny = (source.y + random.randint(-3, 3)) % H
            seed = Individual(new_sp.id, source.mutate(self.env_optimum),
                              (nx, ny), self.step_count)
            self._add_individual(seed)

    def _check_extinctions(self) -> None:
        """Mark species with fewer than 3 living members as extinct."""
        counts: dict = defaultdict(int)
        for ind in self.individuals.values():
            counts[ind.species_id] += 1
        for sid, sp in self.species.items():
            if sp.alive and counts.get(sid, 0) < 3:
                sp.mark_extinct(self.step_count)
                self._recently_extinct_colors[sp.color] = self.step_count

    def _record_history(self) -> None:
        """Append a snapshot of the current population state to history."""
        by_species: dict = defaultdict(list)
        for ind in self.individuals.values():
            by_species[ind.species_id].append(ind)

        sp_snap: dict = {}
        for sid, sp in self.species.items():
            if sp.alive:
                members = by_species.get(sid, [])
                if members:
                    means = {t: sum(m.traits[t] for m in members) / len(members)
                             for t in TRAIT_NAMES}
                else:
                    means = dict(sp.mean_traits)
                sp_snap[sid] = {"pop": len(members), "mean_traits": means}

        self.history.append({
            "step":        self.step_count,
            "total_pop":   len(self.individuals),
            "num_species": sum(1 for sp in self.species.values() if sp.alive),
            "env_optimum": self.env_optimum,
            "species":     sp_snap,
        })

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _add_individual(self, ind: Individual) -> None:
        self.individuals[ind.id] = ind
        self.grid.setdefault((ind.x, ind.y), []).append(ind.id)

    def _remove_individual(self, ind_id: int) -> None:
        ind = self.individuals.pop(ind_id, None)
        if ind is None:
            return
        cell = (ind.x, ind.y)
        id_list = self.grid.get(cell, [])
        if ind_id in id_list:
            id_list.remove(ind_id)
        if not id_list and cell in self.grid:
            del self.grid[cell]

    def _log_event(self, event_type: str, ind: Individual) -> None:
        entry: dict = {
            "event_type":    event_type,
            "step":          self.step_count,
            "individual_id": ind.id,
            "species_id":    ind.species_id,
            "x":             ind.x,
            "y":             ind.y,
        }
        entry.update({t: ind.traits[t] for t in TRAIT_NAMES})
        self.event_log.append(entry)

    def _get_color_for_new_species(self) -> tuple:
        """Return a palette colour not currently in use and not recently freed."""
        self._recently_extinct_colors = {
            c: s for c, s in self._recently_extinct_colors.items()
            if self.step_count - s < 20
        }
        in_use = {sp.color for sp in self.species.values() if sp.alive}
        for color in COLOR_PALETTE:
            if color not in in_use and color not in self._recently_extinct_colors:
                return color
        for color in COLOR_PALETTE:
            if color not in in_use:
                return color
        return COLOR_PALETTE[self._species_id_counter % len(COLOR_PALETTE)]

    # ── Query helpers ────────────────────────────────────────────────────────
    def get_living_species_sorted(self) -> list:
        """Living species sorted by current population (largest first)."""
        counts: dict = defaultdict(int)
        for ind in self.individuals.values():
            counts[ind.species_id] += 1
        return sorted(
            [sp for sp in self.species.values() if sp.alive],
            key=lambda s: counts.get(s.id, 0), reverse=True
        )

    def total_population(self) -> int:
        return len(self.individuals)

    def living_species_count(self) -> int:
        return sum(1 for sp in self.species.values() if sp.alive)


# ── Visualizer ────────────────────────────────────────────────────────────────
class Visualizer:
    """
    Real-time Pygame visualisation of the simulation.

    The window is divided into three horizontal bands:
      - Top bar (30 px)   : global status (step, pop, species, speed, env)
      - Grid area (590 px): 80x60 toroidal grid with species shapes and flash FX
      - HUD (180 px)      : scrollable per-species stats table

    A right-side info panel fills the unused horizontal space beside the grid.

    Keyboard / Mouse
    ----------------
    SPACE   pause / resume
    F       cycle speed  1x -> 5x -> 20x -> 100x
    Q       stop and run post-simulation analysis
    Scroll  scroll HUD species table
    """

    # (steps_per_frame, target_fps, display_label)
    # SLOW runs at 8 fps so each step is visible; higher modes use full 60 fps.
    SPEEDS = [
        (1,   1,  "SLOW"),
        (1,  60,  "1x"),
        (5,  60,  "5x"),
        (20, 60,  "20x"),
        (100, 60, "100x"),
    ]

    def __init__(self, engine: SimulationEngine, config: dict) -> None:
        pygame.init()
        pygame.font.init()

        self.engine = engine
        self.config = config
        self.speed_index: int = 0
        self.paused: bool = False
        self.hud_scroll: int = 0

        W = config["window_width"]
        H = config["window_height"]
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Evolution Simulator")
        self.clock = pygame.time.Clock()

        # ── Layout ────────────────────────────────────────────────────────────
        tb  = config["topbar_height"]    # 30 px
        hud = config["hud_height"]       # 180 px
        grid_area_h = H - tb - hud       # 590 px

        gw = config["grid_width"]        # 80 cells
        gh = config["grid_height"]       # 60 cells

        self.cell_size: int   = max(4, grid_area_h // gh)   # 9 px
        self.grid_pixel_w: int = self.cell_size * gw         # 720 px
        self.grid_pixel_h: int = self.cell_size * gh         # 540 px
        self.grid_ox: int = 0
        self.grid_oy: int = tb

        # Right-side panel (480 px wide)
        self.panel_x: int = self.grid_pixel_w + 10
        self.panel_y: int = tb + 10
        self.panel_w: int = W - self.grid_pixel_w - 14

        self.topbar_rect = pygame.Rect(0, 0, W, tb)
        self.hud_rect    = pygame.Rect(0, H - hud, W, hud)
        self.hud_row_h: int = 27

        # ── Fonts ─────────────────────────────────────────────────────────────
        self.font_s = pygame.font.SysFont("consolas", 12)
        self.font_m = pygame.font.SysFont("consolas", 14, bold=True)

        # ── Colour palette ─────────────────────────────────────────────────────
        self.BG        = (10,  10,  20)
        self.TOPBAR_BG = (22,  22,  38)
        self.HUD_BG    = (18,  18,  32)
        self.GRID_LINE = (28,  32,  44)
        self.PANEL_BG  = (16,  20,  36)
        self.TEXT      = (200, 210, 220)
        self.TEXT_DIM  = (110, 120, 135)
        self.SEP       = (55,  60,  80)

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self) -> None:
        """Pygame main event loop.  Runs until the user quits."""
        while True:
            self._handle_events()
            if not self.engine.running:
                break

            cur = self.SPEEDS[self.speed_index]
            if not self.paused:
                for _ in range(cur[0]):
                    self.engine.step()
                    if not self.engine.running:
                        break
            # Always render; at 100x each rendered frame = 100 sim steps
            self._render()
            self.clock.tick(cur[1])

        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.engine.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.engine.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_f:
                    self.speed_index = (self.speed_index + 1) % len(self.SPEEDS)
            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if self.hud_rect.collidepoint(mx, my):
                    self.hud_scroll = max(0, self.hud_scroll - event.y * self.hud_row_h)

    # ── Rendering ─────────────────────────────────────────────────────────────
    def _render(self) -> None:
        self.screen.fill(self.BG)
        self._draw_topbar()
        self._draw_grid_lines()
        self._draw_individuals()
        self._draw_right_panel()
        self._draw_hud()
        pygame.display.flip()

    def _draw_topbar(self) -> None:
        pygame.draw.rect(self.screen, self.TOPBAR_BG, self.topbar_rect)
        state = "[PAUSED]" if self.paused else "[RUNNING]"
        label = self.SPEEDS[self.speed_index][2]
        text = (f"Step: {self.engine.step_count:,}  |  "
                f"Pop: {self.engine.total_population():,}  |  "
                f"Species: {self.engine.living_species_count()}  |  "
                f"Speed: {label}  |  "
                f"Env Opt: {self.engine.env_optimum:.2f}  |  "
                f"{state}")
        surf = self.font_m.render(text, True, self.TEXT)
        tb = self.config["topbar_height"]
        self.screen.blit(surf, (8, (tb - surf.get_height()) // 2))

    def _draw_grid_lines(self) -> None:
        cs = self.cell_size
        ox, oy = self.grid_ox, self.grid_oy
        gw = self.config["grid_width"]
        gh = self.config["grid_height"]
        for col in range(gw + 1):
            x = ox + col * cs
            pygame.draw.line(self.screen, self.GRID_LINE,
                             (x, oy), (x, oy + self.grid_pixel_h))
        for row in range(gh + 1):
            y = oy + row * cs
            pygame.draw.line(self.screen, self.GRID_LINE,
                             (ox, y), (ox + self.grid_pixel_w, y))

    def _draw_individuals(self) -> None:
        cs     = self.cell_size
        radius = max(2, cs // 2 - 1)
        ox, oy = self.grid_ox, self.grid_oy
        for ind in self.engine.individuals.values():
            sp = self.engine.species.get(ind.species_id)
            if sp is None:
                continue
            cx = ox + ind.x * cs + cs // 2
            cy = oy + ind.y * cs + cs // 2
            self._draw_shape(self.screen, sp.shape, sp.color, (cx, cy), radius)

    def _draw_shape(self, surface: pygame.Surface, shape: str,
                    color: tuple, center: tuple, radius: int) -> None:
        """Render one of 6 distinct shapes at the given pixel centre."""
        cx, cy = center
        if shape == "circle":
            pygame.draw.circle(surface, color, (cx, cy), radius)
            return
        n_sides = {"triangle": 3, "square": 4, "diamond": 4,
                   "pentagon": 5, "hexagon": 6}[shape]
        offset  = {"triangle": -math.pi / 2, "square": -math.pi / 4,
                   "diamond":   0.0,          "pentagon": -math.pi / 2,
                   "hexagon":   0.0}[shape]
        pts = [(int(cx + radius * math.cos(2 * math.pi * i / n_sides + offset)),
                int(cy + radius * math.sin(2 * math.pi * i / n_sides + offset)))
               for i in range(n_sides)]
        pygame.draw.polygon(surface, color, pts)

    def _draw_right_panel(self) -> None:
        """Draw the info panel in the blank area to the right of the grid."""
        label = self.SPEEDS[self.speed_index][2]
        lines = [
            "--- Controls ---",
            "SPACE  pause/resume",
            "F      cycle speed",
            "Q      quit+analyze",
            "Scroll scroll HUD",
            "",
            "--- Environment ---",
            f"Optimum: {self.engine.env_optimum:.3f}",
            f"Step:    {self.engine.step_count:,}",
            f"Speed:   {label}",
            "",
            "--- Species ---",
        ]
        living = self.engine.get_living_species_sorted()
        counts: dict = defaultdict(int)
        for ind in self.engine.individuals.values():
            counts[ind.species_id] += 1
        for sp in living[:10]:
            pop = counts.get(sp.id, 0)
            lines.append(f"  #{sp.id}: {sp.shape[:3]}  pop={pop}")

        px, py = self.panel_x, self.panel_y
        line_h = 16
        panel_h = len(lines) * line_h + 8
        pygame.draw.rect(self.screen, self.PANEL_BG,
                         pygame.Rect(px - 4, py - 4, self.panel_w, panel_h))
        for i, line in enumerate(lines):
            col = self.TEXT_DIM if line.startswith("---") else self.TEXT
            surf = self.font_s.render(line, True, col)
            self.screen.blit(surf, (px, py + i * line_h))

    def _draw_hud(self) -> None:
        """Scrollable HUD showing per-species stats, sorted by population."""
        H   = self.config["window_height"]
        W   = self.config["window_width"]
        hud_h = self.config["hud_height"]
        hy  = H - hud_h

        pygame.draw.rect(self.screen, self.HUD_BG, self.hud_rect)
        pygame.draw.line(self.screen, self.SEP, (0, hy), (W, hy), 2)

        hdr = " Icon  Sp#    Pop    rep_age  surv   rep    mut    env"
        self.screen.blit(self.font_s.render(hdr, True, self.TEXT_DIM), (4, hy + 3))

        living = self.engine.get_living_species_sorted()
        counts: dict = defaultdict(int)
        for ind in self.engine.individuals.values():
            counts[ind.species_id] += 1

        content_y = hy + 20
        content_h = hud_h - 20
        row_h = self.hud_row_h

        max_scroll = max(0, len(living) * row_h - content_h)
        self.hud_scroll = min(self.hud_scroll, max_scroll)

        clip = pygame.Rect(0, content_y, W, content_h)
        prev_clip = self.screen.get_clip()
        self.screen.set_clip(clip)

        for i, sp in enumerate(living):
            row_y = content_y + i * row_h - self.hud_scroll
            if row_y + row_h < content_y or row_y > H:
                continue

            icon = self._render_shape_icon(sp.shape, sp.color)
            self.screen.blit(icon, (4, row_y + (row_h - icon.get_height()) // 2))

            pop = counts.get(sp.id, 0)
            mt  = sp.mean_traits
            row_text = (
                f"      #{sp.id:<4d}  {pop:>4d}   "
                f"{mt['rep_age']:>6.1f}   {mt['survivability']:.2f}  "
                f"{mt['rep_rate']:.2f}  {mt['mutation_rate']:.2f}  "
                f"{mt['env_score']:.2f}"
            )
            alt = self.TEXT if i % 2 == 0 else (175, 185, 200)
            self.screen.blit(
                self.font_s.render(row_text, True, alt),
                (4, row_y + (row_h - self.font_s.get_height()) // 2)
            )

        self.screen.set_clip(prev_clip)

    def _render_shape_icon(self, shape: str, color: tuple,
                            size: int = 14) -> pygame.Surface:
        """Create a small transparent surface with the species shape drawn on it."""
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        r = max(1, size // 2 - 1)
        self._draw_shape(surf, shape, color, (size // 2, size // 2), r)
        return surf


# ── Analyzer ──────────────────────────────────────────────────────────────────
class Analyzer:
    """
    Post-simulation analysis engine.

    Generates five Matplotlib figures (each saved as a PNG) and three CSV
    exports from the history and event log captured by SimulationEngine.

    Charts produced
    ---------------
    1. population_over_time.png   — stacked area chart per species
    2. trait_timelines.png        — 5 subplots of pop-weighted mean traits
    3. diversity_index.png        — Shannon entropy over time
    4. species_summary.png        — pie chart (final pop) + species count line
    5. fitness_landscape.png      — scatter surv vs rep_rate + theory curve
    """

    def __init__(self, engine: SimulationEngine) -> None:
        self.engine      = engine
        self.history     = engine.history
        self.species     = engine.species
        self.individuals = engine.individuals
        self.event_log   = engine.event_log
        self.step_count  = engine.step_count

    def run(self) -> None:
        """Export CSVs and display all analysis charts."""
        print("\n[Analyzer] Exporting CSVs ...")
        self._export_species_history_csv()
        self._export_individual_events_csv()
        self._export_simulation_summary_csv()

        print("[Analyzer] Generating charts ...")
        self._plot_stacked_population()
        self._plot_trait_timelines()
        self._plot_shannon_entropy()
        self._plot_species_summary()
        self._plot_survivability_vs_reprate()

        print("[Analyzer] Done — close chart windows to exit.")
        plt.show()

    # ── CSV exports ───────────────────────────────────────────────────────────
    def _export_species_history_csv(self) -> None:
        rows = []
        for snap in self.history:
            for sid, data in snap["species"].items():
                mt = data["mean_traits"]
                rows.append({
                    "species_id":         sid,
                    "step":               snap["step"],
                    "mean_rep_age":       mt["rep_age"],
                    "mean_survivability": mt["survivability"],
                    "mean_rep_rate":      mt["rep_rate"],
                    "mean_mutation_rate": mt["mutation_rate"],
                    "mean_env_score":     mt["env_score"],
                    "population":         data["pop"],
                    "extinct":            not self.species[sid].alive,
                })
        pd.DataFrame(rows).to_csv("species_history.csv", index=False)
        print("  species_history.csv")

    def _export_individual_events_csv(self) -> None:
        pd.DataFrame(self.event_log).to_csv("individual_events.csv", index=False)
        print("  individual_events.csv")

    def _export_simulation_summary_csv(self) -> None:
        rows = []
        for sp in self.species.values():
            mt = sp.mean_traits
            rows.append({
                "species_id":                sp.id,
                "parent_id":                 sp.parent_id,
                "survived_to_end":           sp.alive,
                "mean_rep_age":              mt["rep_age"],
                "mean_survivability":        mt["survivability"],
                "mean_rep_rate":             mt["rep_rate"],
                "mean_mutation_rate":        mt["mutation_rate"],
                "mean_env_score":            mt["env_score"],
                "lifespan_steps":            sp.lifespan_steps(self.step_count),
                "peak_population":           sp.peak_population,
                "speciation_events_spawned": sp.speciation_events_spawned,
            })
        pd.DataFrame(rows).to_csv("simulation_summary.csv", index=False)
        print("  simulation_summary.csv")

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _norm_color(self, rgb: tuple) -> tuple:
        return tuple(c / 255.0 for c in rgb)

    def _build_steps(self) -> list:
        return [s["step"] for s in self.history]

    def _build_timeline(self) -> dict:
        """Returns {species_id: {step: population}} from the recorded history."""
        timeline: dict = {}
        for snap in self.history:
            for sid, data in snap["species"].items():
                timeline.setdefault(sid, {})[snap["step"]] = data["pop"]
        return timeline

    # ── Charts ────────────────────────────────────────────────────────────────
    def _plot_stacked_population(self) -> None:
        """Stacked area chart of population per species over time."""
        steps = self._build_steps()
        if not steps:
            return
        timeline  = self._build_timeline()
        all_sids  = sorted(timeline.keys())
        pop_matrix, colors, labels = [], [], []
        for sid in all_sids:
            sp = self.species[sid]
            pop_matrix.append([timeline[sid].get(s, 0) for s in steps])
            colors.append(self._norm_color(sp.color))
            labels.append(f"Sp#{sid}")

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.stackplot(steps, *pop_matrix, labels=labels, colors=colors, alpha=0.82)
        ax.set(xlabel="Simulation Step", ylabel="Population",
               title="Species Population Over Time (Stacked Area)")
        ax.legend(loc="upper right", fontsize=7, ncol=4)
        fig.tight_layout()
        fig.savefig("population_over_time.png", dpi=150)
        print("  population_over_time.png")

    def _plot_trait_timelines(self) -> None:
        """Five subplots: population-weighted mean of each trait over time."""
        steps = self._build_steps()
        if not steps:
            return
        trait_data: dict = {t: [] for t in TRAIT_NAMES}
        for snap in self.history:
            total = snap["total_pop"]
            if total == 0:
                for t in TRAIT_NAMES:
                    trait_data[t].append(0.0)
                continue
            weighted: dict = {t: 0.0 for t in TRAIT_NAMES}
            for data in snap["species"].values():
                for t in TRAIT_NAMES:
                    weighted[t] += data["mean_traits"][t] * data["pop"]
            for t in TRAIT_NAMES:
                trait_data[t].append(weighted[t] / total)

        cmap = {"rep_age": "#3498db", "survivability": "#2ecc71",
                "rep_rate": "#e74c3c", "mutation_rate": "#f39c12",
                "env_score": "#9b59b6"}
        fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
        for i, trait in enumerate(TRAIT_NAMES):
            axes[i].plot(steps, trait_data[trait], color=cmap[trait], linewidth=1.5)
            axes[i].set_ylabel(trait, fontsize=9)
            axes[i].grid(True, alpha=0.3)
        axes[-1].set_xlabel("Simulation Step")
        fig.suptitle("Population-Weighted Mean Traits Over Time", fontsize=12)
        fig.tight_layout()
        fig.savefig("trait_timelines.png", dpi=150)
        print("  trait_timelines.png")

    def _plot_shannon_entropy(self) -> None:
        """Shannon diversity index of species populations over time."""
        steps     = self._build_steps()
        entropies = []
        for snap in self.history:
            total = snap["total_pop"]
            if total == 0:
                entropies.append(0.0)
                continue
            h = 0.0
            for data in snap["species"].values():
                p = data["pop"] / total
                if p > 0:
                    h -= p * math.log2(p)
            entropies.append(h)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(steps, entropies, color="#1abc9c", linewidth=1.5)
        ax.fill_between(steps, entropies, alpha=0.2, color="#1abc9c")
        ax.set(xlabel="Simulation Step", ylabel="Shannon Entropy (bits)",
               title="Species Diversity Index Over Time")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("diversity_index.png", dpi=150)
        print("  diversity_index.png")

    def _plot_species_summary(self) -> None:
        """Pie chart of final population + time-series of species count."""
        fig, (ax_pie, ax_count) = plt.subplots(1, 2, figsize=(14, 5))

        # Final population pie (living species only)
        living = [(sp, sum(1 for ind in self.individuals.values()
                           if ind.species_id == sp.id))
                  for sp in self.species.values() if sp.alive]
        if living:
            labels     = [f"Sp#{sp.id}" for sp, _ in living]
            sizes      = [pop for _, pop in living]
            pie_colors = [self._norm_color(sp.color) for sp, _ in living]
            ax_pie.pie(sizes, labels=labels, colors=pie_colors,
                       autopct="%1.1f%%", startangle=140,
                       textprops={"fontsize": 8})
        ax_pie.set_title("Final Population by Species")

        # Species count over time
        steps  = self._build_steps()
        counts = [s["num_species"] for s in self.history]
        ax_count.plot(steps, counts, color="#e67e22", linewidth=1.5)
        ax_count.fill_between(steps, counts, alpha=0.2, color="#e67e22")
        ax_count.set(xlabel="Simulation Step", ylabel="# Living Species",
                     title="Number of Active Species Over Time")
        ax_count.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("species_summary.png", dpi=150)
        print("  species_summary.png")

    def _plot_survivability_vs_reprate(self) -> None:
        """Scatter of current individuals (surv vs rep_rate) + theory curve."""
        fig, ax = plt.subplots(figsize=(10, 7))

        by_species: dict = {}
        for ind in self.individuals.values():
            sid = ind.species_id
            if sid not in by_species:
                by_species[sid] = {"survs": [], "reps": []}
            by_species[sid]["survs"].append(ind.traits["survivability"])
            by_species[sid]["reps"].append(ind.traits["rep_rate"])

        for sid, data in by_species.items():
            sp = self.species.get(sid)
            if sp is None or not data["survs"]:
                continue
            color = self._norm_color(sp.color)
            ax.scatter(data["survs"], data["reps"],
                       c=[color] * len(data["survs"]),
                       alpha=0.5, s=12, label=f"Sp#{sid}")

        # Theoretical coupling curve
        k     = self.engine.config["k_coupling"]
        alpha = self.engine.config["alpha_coupling"]
        s_range  = np.linspace(0.01, 1.0, 300)
        rr_curve = np.clip(k / s_range ** alpha, 0.05, 8.0)
        ax.plot(s_range, rr_curve, "k--", linewidth=2,
                label=f"Theory  k/s^{alpha}  (k={k})")
        ax.set(xlabel="Survivability", ylabel="Replication Rate",
               title="Fitness Landscape: Survivability vs Replication Rate")
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("fitness_landscape.png", dpi=150)
        print("  fitness_landscape.png")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = SimulationEngine(CONFIG)
    vis    = Visualizer(engine, CONFIG)
    vis.run()

    print("\nSimulation ended.  Running post-simulation analysis ...")
    analyzer = Analyzer(engine)
    analyzer.run()
