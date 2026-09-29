# TrustAgriLLM Complete Implementation

## Implemented components

- EfficientNet-B0 plant disease assessment
- Distributed farm/client simulation
- Local model training
- Federated Averaging (FedAvg)
- Model-update hashing
- Permissioned blockchain-style provenance ledger
- Trust-aware update selection
- Malicious client simulation
- Disease performance metrics
- Blockchain validation latency measurement
- Multimodal context construction
- Multimodal Large Language Model reasoning using Qwen2.5-VL

## Dataset

Place an ImageFolder plant disease dataset here:

data/plant_disease/
    Disease_Class_1/
    Disease_Class_2/
    ...

## Installation

pip install -r requirements.txt

## Run

Prepare distributed clients:

python trustagrillm.py prepare

Run normal federated experiment:

python trustagrillm.py federated

Run malicious-client trust experiment:

python trustagrillm.py federated --malicious_clients 1

View results:

python trustagrillm.py summary

Run MLLM:

python trustagrillm.py mllm --image path/to/leaf.jpg

## Important research note

The MLLM output should be evaluated using a documented evaluation protocol.
A human-evaluation CSV template is included in the project.
