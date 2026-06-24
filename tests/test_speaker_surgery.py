import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.finetune.add_speaker import expand_speaker


def test_speaker_surgery_preserves_existing_weights():
    config = {
        "model": {"n_speakers": 2},
        "speakers": {"alpha": 0, "beta": 1},
    }
    embedding = torch.arange(8, dtype=torch.float16).reshape(2, 4)
    shared_weight = torch.randn(3, 3, dtype=torch.float16)
    checkpoint = {
        "generator": {
            "emb_g.weight": embedding.clone(),
            "shared.weight": shared_weight.clone(),
        },
        "step": 123,
    }
    original_config = copy.deepcopy(config)
    original_checkpoint = copy.deepcopy(checkpoint)

    expanded_config, expanded_checkpoint, key = expand_speaker(
        config, checkpoint, "beta", "gamma"
    )

    assert config == original_config
    assert torch.equal(checkpoint["generator"][key], original_checkpoint["generator"][key])
    assert expanded_config["speakers"] == {"alpha": 0, "beta": 1, "gamma": 2}
    assert expanded_config["model"]["n_speakers"] == 3
    assert expanded_checkpoint["generator"][key].dtype == torch.float16
    assert torch.equal(expanded_checkpoint["generator"][key][:2], embedding)
    assert torch.equal(expanded_checkpoint["generator"][key][2], embedding[1])
    assert torch.equal(
        expanded_checkpoint["generator"]["shared.weight"], shared_weight
    )


if __name__ == "__main__":
    test_speaker_surgery_preserves_existing_weights()
    print("Speaker surgery tests: OK")
