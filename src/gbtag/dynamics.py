"""Evolutionary dynamics over the space of identity designs.

Both dynamics are driven by the *private* functional ``pi_P``, never by the
social one: a design spreads because operators adopt it, and an operator
compares designs by what it receives.

* the deterministic replicator equation in an infinite population, used for the
  attractor and bifurcation analysis;
* the finite-population pairwise-comparison (Fermi) process, used for the
  stationary-distribution analysis.  The joint design space is
  ``3 x 4 x 4 = 48`` designs, so the full state space is intractable for the
  sizes of interest and the small-mutation limit is the default route; the full
  chain is used on reduced spaces to check it.

The replicator flow is evaluated with EGTtools (Fernandez Domingos et al.,
2023).  The finite-population route below is a closed-form birth-death
fixation probability rather than the generic EGTtools routine, which is what
makes the 48-design space tractable; it is checked against EGTtools on
reduced spaces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp

import egttools
from egttools.analytical import PairwiseComparison, replicator_equation
from egttools.games import Matrix2PlayerGameHolder


# --------------------------------------------------------------------------
# infinite-population replicator dynamics
# --------------------------------------------------------------------------


def replicator_field(x: np.ndarray, payoff: np.ndarray) -> np.ndarray:
    """Right-hand side of the replicator equation for ``payoff``."""
    return replicator_equation(
        np.asarray(x, dtype=float), np.asarray(payoff, dtype=float)
    )


def integrate_replicator(
    payoff: np.ndarray,
    x0: np.ndarray,
    t_end: float = 500.0,
    n_points: int = 2001,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the replicator equation from ``x0``.

    Returns ``(times, trajectory)`` with ``trajectory`` of shape
    ``(n_points, n_designs)``.
    """
    payoff = np.asarray(payoff, dtype=float)

    def rhs(_t: float, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, None)
        total = x.sum()
        if total > 0:
            x = x / total
        return replicator_equation(x, payoff)

    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        np.asarray(x0, dtype=float),
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        method="LSODA",
    )
    traj = sol.y.T
    traj = np.clip(traj, 0.0, None)
    traj /= traj.sum(axis=1, keepdims=True)
    return sol.t, traj


def replicator_attractor(
    payoff: np.ndarray, x0: np.ndarray, t_end: float = 3000.0
) -> np.ndarray:
    """End state of the replicator flow started at ``x0``."""
    _, traj = integrate_replicator(payoff, x0, t_end=t_end, n_points=2)
    return traj[-1]


def replicator_attractors(
    payoff: np.ndarray,
    n_starts: int = 200,
    # The published run seeds from config.SEED.  A different default here would
    # silently give a reader who calls this directly a different basin share
    # from the one in the paper, with nothing to point at.
    seed: int = 20260819,
    t_end: float = 3000.0,
) -> np.ndarray:
    """End states of the replicator flow from ``n_starts`` interior starts.

    Returns the whole ``(n_starts, n)`` array rather than a summary.  The
    distinction matters: when the flow is multistable, the arithmetic mean of
    the end states is not itself a state the flow reaches, so any *nonlinear*
    observable has to be evaluated attractor by attractor and averaged
    afterwards.  Every observable in this study is bilinear in the state, so
    evaluating one at the mean is exactly the error of reporting
    ``f(E[x])`` where ``E[f(x)]`` is meant.
    """
    rng = np.random.default_rng(seed)
    n = payoff.shape[0]
    ends = np.empty((n_starts, n))
    for k in range(n_starts):
        ends[k] = replicator_attractor(payoff, rng.dirichlet(np.ones(n)), t_end=t_end)
    return ends


def basin_average(values: np.ndarray) -> float:
    """Average of a scalar observable already evaluated at each attractor."""
    return float(np.mean(np.asarray(values, dtype=float)))


# --------------------------------------------------------------------------
# replicator-mutator dynamics: a genuinely polymorphic deterministic flow
# --------------------------------------------------------------------------


def replicator_mutator_field(
    x: np.ndarray, payoff: np.ndarray, mu: float, shift: float | None = None
) -> np.ndarray:
    r"""Right-hand side of the replicator-mutator equation with uniform mutation.

    .. math::
        \dot{x}_i = (1 - \mu)\, x_i f_i(x) + \frac{\mu}{n}\,\bar{f}(x)
                    - x_i \bar{f}(x),

    with ``f_i = (A x)_i`` and ``\bar{f} = x^\top A x``.  Every design is
    reintroduced at rate ``mu``, so no face of the simplex is invariant and
    the flow has an interior rest point.  This is the dynamic that describes
    a market in which designs genuinely coexist, as opposed to a multistable
    flow whose separate attractors have been averaged.

    Mutation makes the equation sensitive to an additive constant in the
    payoff, unlike the plain replicator equation, so the payoff is shifted to
    be strictly positive before use.  ``shift`` overrides the automatic
    choice; the reported results are checked against a range of shifts.
    """
    x = np.asarray(x, dtype=float)
    a = np.asarray(payoff, dtype=float)
    if shift is None:
        shift = max(0.0, -a.min()) + 1.0
    a = a + shift
    n = a.shape[0]
    fitness = a @ x
    mean_fitness = float(x @ fitness)
    return (1.0 - mu) * x * fitness + (mu / n) * mean_fitness - x * mean_fitness


def replicator_mutator_equilibrium(
    payoff: np.ndarray,
    mu: float = 0.01,
    shift: float | None = None,
    t_end: float = 20000.0,
    rtol: float = 1e-11,
    atol: float = 1e-13,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Interior rest point of the replicator-mutator flow.

    Integrated from the uniform state unless ``x0`` is given.  For the
    mutation rates used here the rest point is unique and is reached from
    every interior start, which :func:`replicator_mutator_is_unique` checks.
    """
    a = np.asarray(payoff, dtype=float)
    n = a.shape[0]
    if x0 is None:
        x0 = np.full(n, 1.0 / n)

    def rhs(_t: float, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, None)
        total = x.sum()
        if total > 0:
            x = x / total
        return replicator_mutator_field(x, a, mu, shift)

    sol = solve_ivp(
        rhs, (0.0, t_end), np.asarray(x0, dtype=float),
        t_eval=[t_end], rtol=rtol, atol=atol, method="LSODA",
    )
    x = np.clip(sol.y[:, -1], 0.0, None)
    return x / x.sum()


def replicator_mutator_is_unique(
    payoff: np.ndarray,
    mu: float = 0.01,
    n_starts: int = 24,
    seed: int = 20260819,
    tol: float = 1e-6,
    shift: float | None = None,
) -> bool:
    """Whether every interior start reaches the same rest point."""
    rng = np.random.default_rng(seed)
    n = payoff.shape[0]
    reference = replicator_mutator_equilibrium(payoff, mu, shift)
    for _ in range(n_starts):
        other = replicator_mutator_equilibrium(
            payoff, mu, shift, x0=rng.dirichlet(np.ones(n))
        )
        if np.abs(other - reference).max() > tol:
            return False
    return True


# --------------------------------------------------------------------------
# equilibrium notions in the symmetric two-player game
# --------------------------------------------------------------------------


def strict_nash_strategies(payoff: np.ndarray, tol: float = 1e-9) -> list[int]:
    """Indices of pure designs that are strict symmetric Nash equilibria."""
    n = payoff.shape[0]
    return [
        i
        for i in range(n)
        if all(payoff[i, i] > payoff[j, i] + tol for j in range(n) if j != i)
    ]


def neutrally_stable_strategies(payoff: np.ndarray, tol: float = 1e-9) -> list[int]:
    """Indices of pure designs that are neutrally stable.

    ``i`` is neutrally stable if it is a symmetric Nash equilibrium and, for
    every alternative best reply ``j``, it does at least as well against ``j``
    as ``j`` does against itself.
    """
    n = payoff.shape[0]
    out = []
    for i in range(n):
        if any(payoff[j, i] > payoff[i, i] + tol for j in range(n) if j != i):
            continue
        ok = True
        for j in range(n):
            if j == i or payoff[j, i] < payoff[i, i] - tol:
                continue
            if payoff[i, j] < payoff[j, j] - tol:
                ok = False
                break
        if ok:
            out.append(i)
    return out


def invades(payoff: np.ndarray, invader: int, resident: int, tol: float = 1e-9) -> bool:
    """Whether a rare ``invader`` has a selective advantage against ``resident``."""
    return bool(payoff[invader, resident] > payoff[resident, resident] + tol)


# --------------------------------------------------------------------------
# finite-population pairwise-comparison process
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StationaryResult:
    """Stationary regime of the finite-population process."""

    state_distribution: np.ndarray
    """Probability of each population state."""

    strategy_frequencies: np.ndarray
    """Average frequency of each design in the stationary regime."""

    unsafe_frequency: float
    """Stationary population-level Unsafe frequency."""

    population_size: int
    beta: float
    mu: float


def sparse_stationary_distribution(
    transitions: sp.spmatrix, tol: float = 1e-12, max_power_iter: int = 200_000
) -> np.ndarray:
    """Stationary distribution of a row-stochastic sparse transition matrix.

    The singular system ``(P^T - I) pi = 0`` is solved with the last equation
    replaced by the normalisation, with power iteration as a fallback when the
    direct solve is ill-conditioned.
    """
    p = sp.csr_matrix(transitions, dtype=float)
    n = p.shape[0]
    a = (p.transpose() - sp.identity(n, format="csr")).tolil()
    a[n - 1, :] = 1.0
    b = np.zeros(n)
    b[n - 1] = 1.0

    try:
        pi = spla.spsolve(a.tocsc(), b)
        if np.all(np.isfinite(pi)) and pi.sum() > 0:
            pi = np.clip(pi, 0.0, None)
            pi /= pi.sum()
            if np.abs(pi @ p - pi).max() < 1e-8:
                return pi
    except Exception:  # pragma: no cover - direct solve is the fast path
        pass

    pi = np.full(n, 1.0 / n)
    for _ in range(max_power_iter):
        nxt = pi @ p
        nxt /= nxt.sum()
        if np.abs(nxt - pi).max() < tol:
            pi = nxt
            break
        pi = nxt
    return pi


def _state_matrix(population_size: int, nb_strategies: int) -> np.ndarray:
    """Counts of every population state, shape ``(nb_states, nb_strategies)``."""
    nb_states = egttools.calculate_nb_states(population_size, nb_strategies)
    states = np.empty((nb_states, nb_strategies), dtype=float)
    for s in range(nb_states):
        states[s] = egttools.sample_simplex(s, population_size, nb_strategies)
    return states


def _finite_population_average(states: np.ndarray, u: np.ndarray, Z: int) -> np.ndarray:
    """Value of a bilinear observable in every population state.

    For a state with counts ``k``, the probability that a random ordered pair
    of distinct individuals uses designs ``(i, j)`` is
    ``k_i (k_j - delta_ij) / (Z (Z - 1))``.
    """
    k = states
    outer = k[:, :, None] * k[:, None, :]
    idx = np.arange(k.shape[1])
    outer[:, idx, idx] -= k
    return (outer * u[None, :, :]).sum(axis=(1, 2)) / (Z * (Z - 1))


def stationary_analysis(
    payoff: np.ndarray,
    unsafe_frequency: np.ndarray,
    population_size: int = 50,
    beta: float = 1.0,
    mu: float | None = None,
) -> StationaryResult:
    """Stationary distribution of the Fermi process driven by ``payoff``.

    Feasible only for small design spaces; use
    :func:`stationary_analysis_sml` for the full ``(badge, s_in, s_out)`` space.
    """
    payoff = np.ascontiguousarray(np.asarray(payoff, dtype=float))
    nb_strategies = payoff.shape[0]
    if mu is None:
        mu = 1.0 / population_size

    game = Matrix2PlayerGameHolder(nb_strategies, payoff)
    evolver = PairwiseComparison(population_size, game)
    transitions = evolver.calculate_transition_matrix(beta=beta, mu=mu)
    sd = sparse_stationary_distribution(transitions)

    states = _state_matrix(population_size, nb_strategies)
    freqs = (sd[:, None] * states).sum(axis=0) / population_size
    unsafe_by_state = _finite_population_average(
        states, np.asarray(unsafe_frequency, dtype=float), population_size
    )
    return StationaryResult(
        state_distribution=sd,
        strategy_frequencies=freqs,
        unsafe_frequency=float(sd @ unsafe_by_state),
        population_size=population_size,
        beta=beta,
        mu=float(mu),
    )


def fixation_probability(
    payoff: np.ndarray,
    invader: int,
    resident: int,
    population_size: int = 100,
    beta: float = 0.1,
) -> float:
    """Fixation probability of one ``invader`` in a ``resident`` population.

    Closed form of the birth-death chain of the Fermi process.  With
    ``k`` invaders the expected payoffs of the two designs, sampling opponents
    without replacement, are

    .. math::
        \\pi_I(k) &= \\frac{k-1}{Z-1} a_{II} + \\frac{Z-k}{Z-1} a_{IR}, \\\\
        \\pi_R(k) &= \\frac{k}{Z-1} a_{RI} + \\frac{Z-k-1}{Z-1} a_{RR},

    and the ratio of the down and up rates is ``exp(-beta (pi_I - pi_R))``,
    which gives

    .. math::
        \\rho = \\Big[1 + \\sum_{m=1}^{Z-1}
                 \\exp\\big(-\\beta \\sum_{k=1}^{m} (\\pi_I(k) - \\pi_R(k))\\big)\\Big]^{-1}.

    Computing this directly rather than through the generic EGTtools routine is
    what makes the joint ``(badge, s_in, s_out)`` space tractable: the generic route
    is organised around the full population state space, whose size grows like
    ``C(Z + n - 1, n - 1)`` and is astronomical for the ``n`` used here.
    """
    a = np.asarray(payoff, dtype=float)
    z = int(population_size)
    k = np.arange(1, z, dtype=float)
    pi_inv = ((k - 1.0) * a[invader, invader] + (z - k) * a[invader, resident]) / (z - 1.0)
    pi_res = (k * a[resident, invader] + (z - k - 1.0) * a[resident, resident]) / (z - 1.0)
    exponent = -beta * np.cumsum(pi_inv - pi_res)
    # the sum is dominated by its largest term; factoring it out keeps the
    # exponentials in range for the large beta * Z products used in the sweeps
    shift = max(0.0, float(exponent.max()))
    total = float(np.exp(-shift) + np.exp(exponent - shift).sum())
    return float(np.exp(-shift) / total)


def sml_transition_matrix(
    payoff: np.ndarray, population_size: int = 100, beta: float = 0.1
) -> np.ndarray:
    """Embedded chain over the pure designs in the small-mutation limit.

    A single mutant of a uniformly chosen design either fixes or is lost before
    the next mutation arrives, so the population is monomorphic almost always
    and the process reduces to a chain on the ``n`` pure designs with
    ``P[j, i] = rho_{i -> j} / (n - 1)``.
    """
    a = np.asarray(payoff, dtype=float)
    n = a.shape[0]
    p = np.zeros((n, n))
    for resident in range(n):
        for invader in range(n):
            if invader == resident:
                continue
            p[resident, invader] = (
                fixation_probability(a, invader, resident, population_size, beta) / (n - 1)
            )
        p[resident, resident] = 1.0 - p[resident].sum()
    return p


def stationary_analysis_sml(
    payoff: np.ndarray,
    unsafe_frequency: np.ndarray,
    population_size: int = 100,
    beta: float = 0.1,
) -> StationaryResult:
    """Stationary distribution in the small-mutation limit.

    Under rare mutation the population is monomorphic almost always, so the
    stationary distribution of the embedded chain *is* the long-run frequency
    of each design.  This is the tractable route for the full
    ``(badge, s_in, s_out)`` space; :func:`stationary_analysis` checks it against
    the full chain on reduced spaces.
    """
    transitions = sml_transition_matrix(payoff, population_size, beta)
    sd = sparse_stationary_distribution(sp.csr_matrix(transitions))
    u = np.asarray(unsafe_frequency, dtype=float)
    return StationaryResult(
        state_distribution=sd,
        strategy_frequencies=sd,
        unsafe_frequency=float(sd @ np.diag(u)),
        population_size=population_size,
        beta=beta,
        mu=0.0,
    )
