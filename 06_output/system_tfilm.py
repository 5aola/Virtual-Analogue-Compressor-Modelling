"""Lightning system for the GR-conditioned output-transformer LSTM.

Identical loss / metrics / stateful-TBPTT bookkeeping to
``OutputTransformerSystem`` — the only difference is the batch carries an extra
sample-aligned GR curve that is passed to the model as continuous conditioning
(``model_tfilm.OutputTransformerTFiLMLSTM``).
"""

from __future__ import annotations

from system import OutputTransformerSystem, _detach_state, _as_bool


class OutputTransformerTFiLMSystem(OutputTransformerSystem):
    def _run_model(self, batch, phase: str):
        x, gr, wet, mask, is_reset = batch[:5]
        is_reset = _as_bool(is_reset)
        if is_reset:
            self.reset_phase_state(phase)
        pred, new_state = self.model(
            x, gr, self._states[phase], return_state=True
        )  # [B,1,S]
        self._states[phase] = _detach_state(new_state)
        return pred, wet, mask, is_reset
