# zkProbe

- `train_probe.py`: Python file for training a linear probe on internal activations of an LLM from datasets of harmless and harmful prompts.
- `methods/guest/src/main.rs`: Rust file for applying the linear probe on input of activations to classify as harmless or harmful.
- `host/src/main.rs`: Rust file for communicating input to the guest.
