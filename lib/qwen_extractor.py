"""Frozen Qwen audio tower and the paper's packed-batch embedding protocol."""
import json
from pathlib import Path
import numpy as np

class QwenReplay:
    """@brief Frozen audio-tower wrapper for Qwen feature extraction.
    The caller controls companion batches or uses singleton independent calls.
    Only the audio tower is loaded, and all inference runs without gradients.
    """
    def __init__(self, model_path):
        """@brief Load only the frozen Qwen audio tower from local model shards.
        @param model_path Directory containing the Qwen3-Omni config, processor and weights.
        @details Initialize the CUDA/bfloat16 tower and Mel feature extractor.
        @exception RuntimeError CUDA is unavailable.
        @note Medium float32 matmul precision and SDPA match the paper extraction.
              No fine-tuning or text-generation model is used here.
        """
        import torch
        from safetensors import safe_open
        from transformers import Qwen3OmniMoeProcessor
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

        if not torch.cuda.is_available():
            raise RuntimeError('Paper batch replay requires CUDA with bfloat16 support.')
        # Original generate_prembeds.py (and its train import) enables this.
        # Omitting it changes four otherwise identical extraction batches.
        torch.set_float32_matmul_precision('medium')
        model_path = Path(model_path).expanduser()
        cfg = json.loads((model_path / 'config.json').read_text())
        audio_cfg = Qwen3OmniMoeAudioEncoderConfig(**cfg['thinker_config']['audio_config'])
        audio_cfg._attn_implementation = 'sdpa'
        model = Qwen3OmniMoeAudioEncoder(audio_cfg)
        index = json.loads((model_path / 'model.safetensors.index.json').read_text())
        prefix = 'thinker.audio_tower.'
        state = {}
        for shard in sorted({v for k, v in index['weight_map'].items() if k.startswith(prefix)}):
            with safe_open(model_path / shard, framework='pt', device='cpu') as tensors:
                for key in tensors.keys():
                    if key.startswith(prefix):
                        state[key[len(prefix):]] = tensors.get_tensor(key)
        model.load_state_dict(state, strict=True)
        self.model = model.to('cuda', torch.bfloat16).eval()
        self.fe = Qwen3OmniMoeProcessor.from_pretrained(model_path).feature_extractor

    def embed(self, waves):
        """@brief Embed supplied waveforms and restore their original input order.
        @param waves List of mono, 16 kHz, peak-normalized float32 waveforms.
        @return Float32 matrix (len(waves), 2048), one temporal mean per clip.
        @details Stable-sort by length, compute padded Mel features, pack valid frames,
                 encode, split hidden states by downsampled lengths, then undo the sort.
                 Match the original extractor's handling of a final chunk remainder.
        @note The paper's packed SDPA path mixes clip context. Independent mode
              calls this function with one clip at a time; changing companions can
              change outputs in paper mode.
        """
        import torch
        order = np.argsort([-len(w) for w in waves], kind='stable')
        features = self.fe([waves[j] for j in order], sampling_rate=16000,
                           return_tensors='pt', truncation=False,
                           return_attention_mask=True, padding=True)
        x = features['input_features'].to('cuda', torch.bfloat16)
        mask = features['attention_mask'].long().cuda()
        lengths = mask.sum(1)
        packed = x.permute(0, 2, 1)[mask.bool()].permute(1, 0)
        post = [((length - 1) // 2 + 1 - 1) // 2 + 1 for length in lengths.tolist()]
        post = [(length - 1) // 2 + 1 for length in post]
        with torch.no_grad():
            hidden = self.model(packed, feature_lens=lengths).last_hidden_state.float()
        # Match the original extraction's handling of the final chunk.
        post[-1] += hidden.size(0) - sum(post)
        out = np.empty((len(waves), hidden.size(1)), np.float32)
        for row, segment in zip(order, torch.split(hidden, post)):
            out[row] = segment.mean(0).cpu().numpy()
        return out

