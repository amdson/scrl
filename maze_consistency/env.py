"""Maze generation, deterministic dynamics, locked-door variant, random-walk data.

Conventions
-----------
* Cells are flattened row-major: ``cell = row * W + col``.
* Actions: 0=U, 1=D, 2=L, 3=R. Moving into a wall (or a locked door) = stay.
* Episode return R = -L if the goal is reached after L steps, else -(T+1).
  Bins: ``bin = R + T + 1`` in ``[0, T+1]``, so ``K = T + 2``.
* Door flag (per episode, latent): the door is locked with prob ``p_locked``.
  The agent learns the flag only by trying to enter the door cell:
  FLAG_UNKNOWN -> FLAG_LOCKED (bumped) or FLAG_OPEN (entered).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

WALL, OPEN, START, GOAL, DOOR = 0, 1, 2, 3, 4
CELL_CHARS = "#.SGD"
ACTION_NAMES = ["U", "D", "L", "R"]
ACTION_ARROWS = ["↑", "↓", "←", "→"]
DELTAS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)
FLAG_UNKNOWN, FLAG_OPEN, FLAG_LOCKED = 0, 1, 2
N_ACTIONS = 4


def _next_table(passable: np.ndarray) -> np.ndarray:
    """next_pos[cell, a] for a boolean passable grid. Non-passable cells map to themselves."""
    H, W = passable.shape
    nxt = np.zeros((H * W, N_ACTIONS), dtype=np.int32)
    for r in range(H):
        for c in range(W):
            s = r * W + c
            for a in range(N_ACTIONS):
                nr, nc = r + DELTAS[a, 0], c + DELTAS[a, 1]
                if passable[r, c] and 0 <= nr < H and 0 <= nc < W and passable[nr, nc]:
                    nxt[s, a] = nr * W + nc
                else:
                    nxt[s, a] = s
    return nxt


def _bfs_dist(nxt: np.ndarray, source: int) -> np.ndarray:
    """Shortest-path distance from every cell to ``source`` (undirected moves)."""
    n = nxt.shape[0]
    dist = np.full(n, -1, dtype=np.int32)
    dist[source] = 0
    frontier = [source]
    while frontier:
        new = []
        for s in frontier:
            for a in range(N_ACTIONS):
                s2 = nxt[s, a]
                if dist[s2] < 0:
                    dist[s2] = dist[s] + 1
                    new.append(s2)
        frontier = new
    return dist


@dataclass
class Maze:
    grid: np.ndarray  # int8 [H, W] with values WALL/OPEN/START/GOAL/DOOR
    start: int
    goal: int
    T: int
    door: int = -1
    p_locked: float = 0.0
    name: str = "maze"
    # derived (filled in __post_init__)
    H: int = field(init=False)
    W: int = field(init=False)
    n_cells: int = field(init=False)
    K: int = field(init=False)
    next_open: np.ndarray = field(init=False)
    next_locked: np.ndarray = field(init=False)
    dist_open: np.ndarray = field(init=False)
    d_star: int = field(init=False)

    def __post_init__(self):
        self.grid = np.asarray(self.grid, dtype=np.int8)
        self.H, self.W = self.grid.shape
        self.n_cells = self.H * self.W
        self.K = self.T + 2
        passable = self.grid != WALL
        self.next_open = _next_table(passable)
        locked = passable.copy()
        if self.door >= 0:
            locked.flat[self.door] = False
        self.next_locked = _next_table(locked)
        self.dist_open = _bfs_dist(self.next_open, self.goal)
        self.d_star = int(self.dist_open[self.start])
        assert self.d_star > 0, "goal unreachable from start"
        assert self.T >= self.d_star

    # --- bins -------------------------------------------------------------
    @property
    def has_door(self) -> bool:
        return self.door >= 0

    @property
    def n_token_flags(self) -> int:
        """Flags visible in the pos token: 1 (no door) or 2 (normal / known-locked)."""
        return 2 if self.has_door else 1

    @property
    def n_dp_flags(self) -> int:
        return 3 if self.has_door else 1

    @property
    def R_max(self) -> int:
        return -self.d_star

    def bin_of(self, R):
        return np.asarray(R) + self.T + 1

    def R_of(self, b):
        return np.asarray(b) - self.T - 1

    @property
    def R_values(self) -> np.ndarray:
        return np.arange(self.K) - self.T - 1

    def rc(self, cell):
        return divmod(int(cell), self.W)

    def open_cells(self) -> np.ndarray:
        return np.flatnonzero(self.grid.reshape(-1) != WALL)

    def ascii(self, path=None, marks=None) -> str:
        rows = []
        g = self.grid
        for r in range(self.H):
            row = []
            for c in range(self.W):
                ch = CELL_CHARS[g[r, c]]
                cell = r * self.W + c
                if marks and cell in marks:
                    ch = marks[cell]
                elif path is not None and cell in path and g[r, c] == OPEN:
                    ch = "*"
                row.append(ch)
            rows.append("".join(row))
        return "\n".join(rows)

    def __repr__(self):
        return (f"Maze({self.name}: {self.H}x{self.W}, d*={self.d_star}, T={self.T}, K={self.K}, "
                f"door={self.door}, p_locked={self.p_locked})")


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def recursive_backtracker(rooms_h: int, rooms_w: int, rng: np.random.Generator) -> np.ndarray:
    H, W = 2 * rooms_h + 1, 2 * rooms_w + 1
    grid = np.full((H, W), WALL, dtype=np.int8)
    visited = np.zeros((rooms_h, rooms_w), dtype=bool)
    r0, c0 = rng.integers(rooms_h), rng.integers(rooms_w)
    stack = [(r0, c0)]
    visited[r0, c0] = True
    grid[2 * r0 + 1, 2 * c0 + 1] = OPEN
    while stack:
        r, c = stack[-1]
        nbrs = [(r + dr, c + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= r + dr < rooms_h and 0 <= c + dc < rooms_w and not visited[r + dr, c + dc]]
        if not nbrs:
            stack.pop()
            continue
        nr, nc = nbrs[rng.integers(len(nbrs))]
        grid[2 * r + 1 + (nr - r), 2 * c + 1 + (nc - c)] = OPEN
        grid[2 * nr + 1, 2 * nc + 1] = OPEN
        visited[nr, nc] = True
        stack.append((nr, nc))
    return grid


def braid(grid: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Remove dead ends with probability p by opening a wall that leads to another open cell."""
    grid = grid.copy()
    H, W = grid.shape
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if grid[r, c] == WALL:
                continue
            open_nbrs = [(r + dr, c + dc) for dr, dc in DELTAS if grid[r + dr, c + dc] != WALL]
            if len(open_nbrs) != 1 or rng.random() > p:
                continue
            cands = []
            for dr, dc in DELTAS:
                wr, wc = r + dr, c + dc
                fr, fc = r + 2 * dr, c + 2 * dc
                if grid[wr, wc] == WALL and 0 < fr < H - 1 and 0 < fc < W - 1 and grid[fr, fc] != WALL:
                    cands.append((wr, wc))
            if cands:
                wr, wc = cands[rng.integers(len(cands))]
                grid[wr, wc] = OPEN
    return grid


def make_maze(rooms=(4, 4), braid_p=0.5, seed=0, T=None, d_range=(6, 12), name=None) -> Maze | None:
    """Fixed-layout maze: backtracker + braiding, start/goal chosen with d* in d_range."""
    rng = np.random.default_rng(seed)
    grid = braid(recursive_backtracker(rooms[0], rooms[1], rng), braid_p, rng)
    passable = grid != WALL
    nxt = _next_table(passable)
    cells = np.flatnonzero(passable.reshape(-1))
    for _ in range(50):
        start = int(rng.choice(cells))
        dist = _bfs_dist(nxt, start)
        goals = [c for c in cells if d_range[0] <= dist[c] <= d_range[1]]
        if goals:
            goal = int(rng.choice(goals))
            g = grid.copy()
            g.flat[start] = START
            g.flat[goal] = GOAL
            T = T if T is not None else int(2.5 * dist[goal])
            return Maze(g, start, goal, T, name=name or f"bt{rooms[0]}x{rooms[1]}_b{braid_p}_s{seed}")
    return None


def maze_from_ascii(text: str, T: int, p_locked: float = 0.0, name="ascii") -> Maze:
    rows = [ln for ln in text.strip("\n").splitlines() if ln.strip()]
    W = max(len(r) for r in rows)
    grid = np.full((len(rows), W), WALL, dtype=np.int8)
    start = goal = door = -1
    for r, ln in enumerate(rows):
        for c, ch in enumerate(ln):
            v = CELL_CHARS.index(ch)
            grid[r, c] = v
            cell = r * W + c
            if v == START:
                start = cell
            elif v == GOAL:
                goal = cell
            elif v == DOOR:
                door = cell
    return Maze(grid, start, goal, T, door=door, p_locked=p_locked if door >= 0 else 0.0, name=name)


# The locked-door maze (H3). Route via door: 4 steps if open. Around: 12.
# If locked: 1 step to the cell before the door, 1 bump, 1 back, then 12 = 15.
DOOR_MAZE_ASCII = """
#########
#S.D.G..#
#.#####.#
#.......#
#########
"""


def door_maze(T=24, p_locked=0.7) -> Maze:
    return maze_from_ascii(DOOR_MAZE_ASCII, T=T, p_locked=p_locked, name="door")


# ---------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------

def step_env(maze: Maze, pos, flag, locked, a):
    """Vectorised one-step transition. Returns (pos', flag', attempted_door)."""
    nxt_open = maze.next_open[pos, a]
    nxt = np.where(locked, maze.next_locked[pos, a], nxt_open)
    attempted = np.zeros_like(pos, dtype=bool)
    if maze.has_door:
        attempted = (nxt_open == maze.door) & (pos != maze.door)
        flag = np.where(attempted & locked, FLAG_LOCKED, flag)
        flag = np.where(nxt == maze.door, FLAG_OPEN, flag)
    return nxt, flag, attempted


def rollout_numpy(maze: Maze, policy, N: int, rng: np.random.Generator, greedy=False):
    """Generic rollout. ``policy(t, pos, flag, locked, history) -> probs [N,4]`` (numpy).

    Returns a dict with positions [N,T+1], flags [N,T+1], actions [N,T], length, returns,
    locked, door_attempted, and per-step behaviour log-prob logq [N,T].
    """
    T = maze.T
    pos = np.full(N, maze.start, dtype=np.int32)
    flag = np.zeros(N, dtype=np.int8)
    locked = rng.random(N) < maze.p_locked
    positions = np.zeros((N, T + 1), dtype=np.int32)
    flags = np.zeros((N, T + 1), dtype=np.int8)
    actions = np.zeros((N, T), dtype=np.int8)
    logq = np.zeros((N, T), dtype=np.float32)
    length = np.full(N, T, dtype=np.int32)
    returns = np.full(N, -(T + 1), dtype=np.int32)
    door_attempted = np.zeros(N, dtype=bool)
    alive = np.ones(N, dtype=bool)
    positions[:, 0] = pos
    for t in range(T):
        probs = policy(t, pos, flag, locked, (positions, flags, actions))
        if greedy:
            a = probs.argmax(-1)
        else:
            cum = np.cumsum(probs, -1)
            u = rng.random(N)[:, None]
            a = np.minimum((u > cum).sum(-1), N_ACTIONS - 1)
        a = a.astype(np.int8)
        lp = np.log(np.maximum(probs[np.arange(N), a], 1e-12))
        nxt, nflag, attempted = step_env(maze, pos, flag, locked, a)
        pos = np.where(alive, nxt, pos)
        flag = np.where(alive, nflag, flag)
        door_attempted |= alive & attempted
        actions[:, t] = a
        logq[:, t] = lp
        positions[:, t + 1] = pos
        flags[:, t + 1] = flag
        reached = alive & (pos == maze.goal)
        length[reached] = t + 1
        returns[reached] = -(t + 1)
        alive &= ~reached
    return dict(positions=positions, flags=flags, actions=actions, length=length, returns=returns,
                locked=locked, door_attempted=door_attempted, logq=logq)


def random_walk_policy(t, pos, flag, locked, hist):
    return np.full((pos.shape[0], N_ACTIONS), 1.0 / N_ACTIONS, dtype=np.float32)


def random_walk_episodes(maze: Maze, N: int, seed: int, drop_optimal: bool = False):
    """Random-walk dataset. If drop_optimal, episodes achieving R = -d* are removed so that the
    optimal return is *never* observed (H2 in its strict form)."""
    rng = np.random.default_rng(seed)
    data = rollout_numpy(maze, random_walk_policy, N, rng)
    if drop_optimal:
        keep = data["returns"] != maze.R_max
        data = {k: v[keep] for k, v in data.items()}
    data["N"] = int(data["returns"].shape[0])
    return data


def dp_state(maze: Maze, pos, flag):
    """DP state index = pos + n_cells * flag (flag in {unknown, open, locked})."""
    if not maze.has_door:
        return np.asarray(pos)
    return np.asarray(pos) + maze.n_cells * np.asarray(flag)


def table_policy(table: np.ndarray, maze: Maze):
    """Wrap a [T, S, 4] DP table as a rollout policy."""
    def policy(t, pos, flag, locked, hist):
        return table[t, dp_state(maze, pos, flag)]
    return policy


def make_open_maze(H=7, W=7, wall_p=0.2, seed=0, T=None, d_range=(6, 12), name=None) -> Maze | None:
    """Open room with random obstacles inside a wall border. Many shortest paths, so the random
    walk has non-negligible probability of an optimal episode (the plan's h0 >= 1e-3 criterion)."""
    rng = np.random.default_rng(seed)
    grid = np.full((H, W), OPEN, dtype=np.int8)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = WALL
    inner = rng.random((H - 2, W - 2)) < wall_p
    grid[1:-1, 1:-1][inner] = WALL
    passable = grid != WALL
    nxt = _next_table(passable)
    cells = np.flatnonzero(passable.reshape(-1))
    # reject disconnected layouts
    if (_bfs_dist(nxt, int(cells[0])) < 0)[cells].any():
        return None
    for _ in range(50):
        start = int(rng.choice(cells))
        dist = _bfs_dist(nxt, start)
        goals = [c for c in cells if d_range[0] <= dist[c] <= d_range[1]]
        if goals:
            goal = int(rng.choice(goals))
            g = grid.copy()
            g.flat[start] = START
            g.flat[goal] = GOAL
            T = T if T is not None else int(2.5 * dist[goal])
            return Maze(g, start, goal, T, name=name or f"open{H}x{W}_p{wall_p}_s{seed}")
    return None


def door_maze_ascii(a=2, b=2, depth=2, wide_around=False):
    """Parametric door layout. Row 1: S, (a-1) cells, D, (b-1) cells, G. The around route drops
    ``depth`` rows at S's column, runs right, and climbs back at G's column.
    open route = a+b ; around = a+b+2*depth ; locked = 2a-1 + around."""
    inner_w = a + b + 1
    W = inner_w + 2
    rows = ["#" * W]
    rows.append("#S" + "." * (a - 1) + "D" + "." * (b - 1) + "G#")
    for _ in range(depth - 1):
        rows.append("#." + "#" * (inner_w - 2) + ".#")
    rows.append("#" + "." * inner_w + "#")
    rows.append("#" * W)
    return "\n".join(rows)


def door_maze_param(a=2, b=2, depth=2, p_locked=0.7, T=None, name=None) -> Maze:
    around = a + b + 2 * depth
    locked = 2 * a - 1 + around
    T = T if T is not None else int(locked + 6)
    return maze_from_ascii(door_maze_ascii(a, b, depth), T=T, p_locked=p_locked,
                           name=name or f"door_a{a}b{b}d{depth}p{p_locked}")


def default_maze(T: int = 20) -> Maze:
    """The fixed maze used by E1/E2/E4/E5 (chosen by scanning seeds for h[0,start,-d*] >= 1e-3)."""
    return make_open_maze(7, 7, 0.15, seed=3, T=T, d_range=(7, 11), name="open7")


def default_door_maze(p_locked: float = 0.7) -> Maze:
    """The locked-door maze used by E3: open route 4, around 6, locked detour 9."""
    return door_maze_param(a=2, b=2, depth=1, p_locked=p_locked, T=15, name="door")


def get_maze(name: str) -> Maze:
    return {"default": default_maze, "door": default_door_maze}[name]()
