# HMoE2 Configuration Reference

This document outlines the structure and available parameters for the `config.yaml` file used to define the Hierarchical Mixture of Experts (HMoE2) topology. The configuration defines the global tasks, the feature routing, and the specific neural backends deployed across the tree.

---

## 1. Root Structure
A valid configuration file must contain two primary root keys: `tasks` and `tree`.

```yaml
tasks:
  # List of global task definitions
tree:
  # Root node of the HMoE architecture (usually a ROUTER)
```

---

## 2. Task Definitions (`tasks`)
Tasks define the predictive objectives of the network. They specify the loss balancing, class weighting, and the ground-truth target.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | String | **Required** | Unique identifier for the task (e.g., `task_regime`). |
| `num_classes` | Integer | **Required** | Number of output classes (e.g., `2` for binary). |
| `loss_weight` | Float | `1.0` | Multiplier for this task's contribution to the total loss. |
| `pos_weight` | Float | `1.0` | Scaling factor applied to the positive class to handle class imbalance. |
| `enabled` | Boolean | `true` | Toggles the task on or off during training/inference. |
| `label_target`| String/Dict | `null` | The ground-truth feature to train against (see *Feature Definitions* below). |

---

## 3. Feature Definitions
Features describe the input streams ingested by Experts or defined as targets in Tasks. They can be provided as a simple string (the feature name) or as an object for advanced control.

**Simple String Format:**
```yaml
- "forward-panel_1d_rsi_14_3"
```

**Advanced Object Format:**
```yaml
- name: "forward-panel_1d_rsi_14_3"
  clamp: 0.0
  normalize: 1
  cheat: false
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | String | **Required** | Exact string matching the feature name in the dataset. |
| `clamp` | Float | `0.0` | Clamping threshold for outliers (if implemented in pre-processing). |
| `normalize` | Integer/Bool| `0` | Flag to apply normalization to this specific feature. |
| `cheat` | Boolean | `false` | If `true`, this feature is silenced/blinded during inference (typically used to mark labels so they aren't used as inputs). |

---

## 4. Node Types

The `tree` is built recursively using two types of nodes: **Routers** and **Experts**.

### 4.1 ROUTER Node
Routers act as traffic controllers, using a gating mechanism to distribute incoming features to their children.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | String | `'unnamed'` | Identifier for the router. |
| `type` | String | **Required** | Must be `ROUTER`. |
| `gate_type` | String | `TCN` | The gating algorithm used (see *Gate Types* below). |
| `noise_std` | Float | `0.1` | Standard deviation of noise injected during training for exploration. |
| `top_k` | Integer | `1` | Number of experts to select (Only applicable if `gate_type` is `TOPK`). |
| `hidden_dim`| Integer | `32` | Hidden dimension size (Only applicable if `gate_type` is `GRU`). |
| `children` | List | `[]` | List of nested child nodes (Routers or Experts). |

### 4.2 EXPERT Node
Experts are the leaf nodes of the tree. They contain the actual neural networks (backends) that process the isolated feature sets and make predictions.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | String | `'unnamed'` | Identifier for the expert. |
| `type` | String | **Required** | Must be `EXPERT`. |
| `backend` | String | `LINEAR` | The neural architecture to deploy (see *Backend Types* below). |
| `hidden_dim`| Integer | `32` | Size of the hidden representation inside the backend. |
| `allowed_tasks`| List[Str]| All | List of task names this expert is allowed to predict. |
| `features` | List | `[]` | The specific features this expert is permitted to see. |
| `dropout` | Float | `0.2` | Universal dropout probability for the backend. |

*(Note: Additional backend-specific parameters like `window_length` or `num_layers` are placed directly in the Expert's configuration block).*

---

## 5. Backend Types (`backend`)
The following neural architectures can be assigned to an Expert.

* **`LINEAR`**: Simple feedforward network. Timesteps are processed independently.
* **`TCN`**: Temporal Convolutional Network. Uses dilated causal convolutions.
    * *Parameters*: `dilations` (List of ints, default: `[1, 2, 4, 8]`).
* **`GRU`**: Gated Recurrent Unit.
    * *Parameters*: `num_layers` (Int, default: `2`).
* **`LSTM`**: Long Short-Term Memory.
    * *Parameters*: `num_layers` (Int, default: `2`).
* **`VANILLA_RNN`** (or `RNN`): Standard Elman Recurrent Neural Network.
    * *Parameters*: `num_layers` (Int, default: `2`).
* **`GATED_RESIDUAL`** (or `GR`): Feedforward network with GLU and residual connections.
* **`CAUSAL_TRANSFORMER`** (or `TRANSFORMER`, `CT`): Transformer encoder with causal masking and sinusoidal embeddings.
    * *Parameters*: `num_layers` (Int, default: `2`), `nheads` (Int, default: `4`).
* **`SIGNATORY`** (or `SIGNATURE`, `ROUGH_PATH`, `RP`): Uses the `signatory` library to compute path signatures over a sliding window.
    * *Parameters*: `depth` (Int, default: `2`), `window_length` (Int, default: `80`).
* **`MATRIX_PROFILE`** (or `MOTIF`, `MP`): Motif-based sequence analysis backend.

---

## 6. Gate Types (`gate_type`)
The routing mechanism assigned to a Router node.

* **`PASS_THROUGH`**: Bypasses competitive gating entirely. All children receive 100% of the payload unconditionally (used for parallel, independent task execution).
* **`LINEAR`**: Stateless routing. Calculates weights strictly based on the current timestep.
* **`TCN`**: Temporal Convolutional Network gate. Context-aware routing over time (Default).
* **`GRU`**: Stateful recurrent gate mapping temporal sequences to routing decisions.
* **`TOPK`**: Sparse routing that zeros out all but the highest `k` confident experts.

---

## 7. Example Configuration

```yaml
tasks:
  - name: "task_regime"
    num_classes: 2
    loss_weight: 1.0
    pos_weight: 4.0
    enabled: true
    label_target:
      name: "target_macro_regime"
      cheat: true

tree:
  name: "root_router"
  type: ROUTER
  gate_type: PASS_THROUGH
  noise_std: 0.0
  children:
    - name: "macro_expert"
      type: EXPERT
      backend: SIGNATORY
      hidden_dim: 64
      depth: 3
      window_length: 200
      dropout: 0.2
      allowed_tasks: 
        - "task_regime"
      features:
        - name: "forward-panel_1W_rsi_14_3"
          normalize: 1
        - name: "forward-panel_1W_macd2_12_26_9_3"
          normalize: 0
```