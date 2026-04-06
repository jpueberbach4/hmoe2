import torch
import torch.nn as nn
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Any

from hmoe2.tensor import HmoeTensor
from hmoe2.heads import HmoeHead

@dataclass
class RegimeStateDTO:
    is_hunting_zone: bool  
    is_chop: bool
    active_neuron_count: int
    smoothed_density: float
    bars_in_current_zone: int
    flipped_this_step: bool

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Any

from hmoe2.tensor import HmoeTensor
from hmoe2.heads import HmoeHead

@dataclass
class RegimeStateDTO:
    is_hunting_zone: bool  
    is_chop: bool
    active_neuron_count: int
    smoothed_density: float
    bars_in_current_zone: int
    flipped_this_step: bool

class RegimeDetector:
    def __init__(
        self, 
        target_module: HmoeHead,
        lookback_window: int = 1000, 
        sma_window: int = 5,           
        entry_debounce: int = 2,       
        exit_debounce: int = 15,       
        high_threshold: float = 0.85,  # STRICT: Requires Bright Yellow (>85% intensity) to start
        low_threshold: float = 0.15    # STRICT: Requires mostly Black (<15% intensity) to end
    ):
        self.target_module = target_module
        self.lookback_window = lookback_window
        self.sma_window = sma_window
        self.entry_debounce = entry_debounce
        self.exit_debounce = exit_debounce
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        
        self._history_buffer = deque(maxlen=lookback_window)
        self._sma_buffer = deque(maxlen=sma_window)
        
        self._is_hunting_zone = False 
        self._bars_in_quiet = 0
        self._bars_in_chaos = 0
        
        self.latest_state: Optional[RegimeStateDTO] = None
        self.state_history: List[RegimeStateDTO] = []
        
        self._hook_handle = self.target_module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module: nn.Module, inputs: tuple, outputs: Any):
        if module.training:
            return
            
        logits = outputs.to_tensor()
        probabilities = torch.softmax(logits, dim=-1)[..., 1]
        
        new_state = self.update(probabilities)
        if new_state is not None:
            self.latest_state = new_state

    def remove_listener(self):
        self._hook_handle.remove()

    def update(self, probabilities: torch.Tensor) -> Optional[RegimeStateDTO]:
        if probabilities.size(0) != 1:
            return None

        probs_seq = probabilities[0].detach().cpu().numpy()
        final_state = None
        
        for i in range(probs_seq.shape[0]):
            raw_prob = float(probs_seq[i])
            
            # ==================================================================
            # PURE MATH: Rolling Causal Normalization
            # We recreate the exact math that made your barcode look so good,
            # so the trading logic triggers on the exact same signal.
            # ==================================================================
            self._history_buffer.append(raw_prob)
            hist_arr = np.array(self._history_buffer)
            roll_min = hist_arr.min()
            roll_max = hist_arr.max()
            roll_range = max(roll_max - roll_min, 1e-8)
            
            # Scale to 0.0 - 1.0, then cube it to isolate the sharp peaks
            norm_signal = ((raw_prob - roll_min) / roll_range) ** 3.0
            
            self._sma_buffer.append(norm_signal)
            smoothed_signal = float(np.mean(self._sma_buffer))
            
            flipped = False
            
            # RULE: Only switch to Uptrend if it hits Bright Yellow
            if smoothed_signal >= self.high_threshold:
                self._bars_in_chaos += 1
                self._bars_in_quiet = 0
                if not self._is_hunting_zone and self._bars_in_chaos >= self.entry_debounce:
                    self._is_hunting_zone = True
                    flipped = True
                    
            # RULE: Only switch to Downtrend if it fades out to Black
            elif smoothed_signal <= self.low_threshold:
                self._bars_in_quiet += 1
                self._bars_in_chaos = 0
                if self._is_hunting_zone and self._bars_in_quiet >= self.exit_debounce:
                    self._is_hunting_zone = False
                    flipped = True
                    
            # HYSTERESIS: If it's a weak Brown line, reset the counters and hold the current trend
            else:
                self._bars_in_quiet = 0
                self._bars_in_chaos = 0

            is_chop = (self.low_threshold < smoothed_signal < self.high_threshold)

            final_state = RegimeStateDTO(
                is_hunting_zone=self._is_hunting_zone,
                is_chop=is_chop,
                active_neuron_count=1 if smoothed_signal > 0.5 else 0, 
                smoothed_density=smoothed_signal,
                bars_in_current_zone=self._bars_in_quiet if not self._is_hunting_zone else self._bars_in_chaos,
                flipped_this_step=flipped
            )
            
            self.state_history.append(final_state)
            
        return final_state