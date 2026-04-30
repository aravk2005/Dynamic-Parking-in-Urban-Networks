# Dynamic Parking in Urban Networks

A reinforcement learning-based intelligent parking recommendation system that dynamically assigns drivers to optimal parking lots in real time, adapting to changing traffic, availability, and pricing conditions.

---

## Problem Overview

A driver traveling from a source to a destination needs to park near the destination. The challenge: conditions change constantly — traffic worsens, parking lots fill up, and prices shift. A static, one-time decision isn't enough.

This system uses a **Deep Q-Network (DQN)** reinforcement learning agent to:
- Continuously evaluate parking options while the driver is en route
- Respect the driver's maximum walking distance constraint
- Minimize total travel time, walking distance, and parking cost ($)

The urban road network is modeled as a **directed graph** `G = (N, E)`, where nodes represent intersections, origin `S`, destination `D`, and parking lots `P = {p₁, p₂, …, pₘ}`. Edge weights `Tᵢⱼ` represent travel times that vary dynamically.

---

## Team

| Member | Task |
|--------|------|
| Anushree | Task 1 – Environment Type Analysis |
| Aleena | Task 2 – PEAS Specification & Agent-Environment Diagram |
| Moussa | Task 3 – Search Problem Formulation |
| Arav & Pedro | Task 4 – Implementation (DQN) |

---

##  Environment Characteristics (Task 1)

| Property | Classification | Reason |
|----------|---------------|--------|
| Observability | **Partially Observable** | Future traffic, availability, and costs are unknown |
| Outcome | **Stochastic** | Routes and parking availability vary unpredictably |
| Decision Structure | **Sequential** | Each action affects future states and options |
| Change over Time | **Dynamic** | Environment evolves independently of the agent |
| State/Action Space | **Discrete** | Finite graph nodes, edges, and parking choices |
| Agents | **Single-Agent** | Optimizes for one driver; other vehicles are environment |

---

## PEAS Specification (Task 2)

**Performance Measure**
- Minimize total driving time from origin to parking lot
- Minimize walking distance from parking lot to destination
- Ensure walking distance ≤ driver's maximum acceptable limit
- Minimize total parking cost
- Avoid recommending frequently-full lots

**Environment**
- Directed graph `G = (N, E)` with dynamic edge weights `Tᵢⱼ(t)`
- Parking costs `C(t) = {c₁, c₂, …, cₙ}` and availability `A(t)` — both time-varying

**Actuators**
- Recommend optimal route to follow
- Select target parking lot
- Trigger rerouting when conditions change significantly
- Commit to a final lot once the driver is close enough

**Sensors**
- Current vehicle position (node in graph)
- Real-time edge travel times `Tᵢⱼ(t)`
- Live parking costs and availability
- Walking distances from each lot to destination
- Driver's maximum walking distance preference
- Current simulation time `t`

---

## Search Problem Formulation (Task 3)

### State
```
s = (v, t, A[t], C[t])
```
- `v` — current node in graph G
- `t` — current time step
- `A[t]` — availability status of each parking lot at time t
- `C[t]` — parking cost of each lot at time t

### Initial State
```
s₀ = (S, t₀, A(t₀), C(t₀))
```

### Actions
- **Navigate**: move along edge `<v, u> ∈ E`
- **Park**: commit to lot `pₖ` if currently at that node and conditions are valid

### Transition Model
```
transition_model(s, a):
  - Drive <v, u>  →  next node = u, t' = t + Tᵥᵤ(t)
  - Update C(t') and A(t') accordingly
  - Park pₖ       →  enter "parked" terminal state (if Goal Test passes)
```

### Goal Test
`Goal_Test(s)` is **true** if:
- Vehicle is at parking lot `pₖ`
- Lot has availability: `Aₖ(t) > 0`
- Walking distance is within limit: `W(pₖ, D) ≤ Wₘₐₓ`

### Path Cost
```
PathCost = Σ driving_time + a · W(pₖ, D) + b · cₖ(t)
```
Where `a` and `b` are tunable weight coefficients balancing walking distance vs. parking cost priority.

---

## Implementation (Task 4)

### Method: Deep Q-Network (DQN)

Built using **PyTorch**, the DQN agent learns to recommend optimal parking lots through trial and error across many simulated trips.

The implementation covers all six search problem components:
1. State Representation
2. Initial State
3. Actions
4. Transition Model
5. Goal Test
6. Path Cost

Training was evaluated at **200, 1,000, and 10,000 episodes** to measure convergence and performance improvement over time.

### Running the Code

```bash
# Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook
```

> Open the `.ipynb` file and run all cells. Adjust the episode count in the training cell as needed.

### Dependencies
- Python 
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

---


## 📊 Results

The DQN agent was trained and tested across increasing episode counts (200 → 1,000 → 10,000), with performance compared against baseline optimization methods covered in class.

---
