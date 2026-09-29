"""
TrustAgriLLM research prototype

Implements:
1) EfficientNet-B0 plant disease classification
2) Distributed local training
3) Trust-aware Federated Averaging (FedAvg)
4) Permissioned blockchain-style provenance ledger
5) Multimodal context construction
6) Qwen2.5-VL agricultural reasoning

Updated features:
- Train / validation / test separation
- Live training progress
- Configurable CPU-friendly experiments
- Per-round result saving
- Trust-aware aggregation
- Blockchain latency measurement
- Client-level experiment results
"""

import argparse
import copy
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from torch.utils.data import (
    DataLoader,
    Subset,
)

from torchvision import datasets, transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

RAW = ROOT / "data" / "plant_disease"
SPLITS = ROOT / "data" / "splits"

RESULTS = ROOT / "results"
LEDGER = ROOT / "ledger"
MODELS = ROOT / "models"

for path in [
    RESULTS,
    LEDGER,
    MODELS,
]:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# DEFAULT EXPERIMENT CONFIGURATION
# ============================================================

SEED = 42

NUM_CLIENTS = 5

# CPU-friendly default configuration
ROUNDS = 3
LOCAL_EPOCHS = 1

BATCH_SIZE = 16

LR = 1e-3

TRUST_THRESHOLD = 0.55

# Data split
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10

# Number of images used per client
# None = use full client dataset
#
# Recommended for CPU testing:
# 500 or 1000
MAX_SAMPLES_PER_CLIENT = 1000

# Number of validation images per client
MAX_VALIDATION_SAMPLES = 300

# Number of final test images
MAX_TEST_SAMPLES = 3000

NUM_WORKERS = 0

# Pretrained model
USE_PRETRAINED = True


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# DATA TRANSFORMS
# ============================================================

def train_transform():

    return transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(10),

        transforms.ToTensor(),
    ])


def eval_transform():

    return transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.ToTensor(),
    ])


# ============================================================
# DATASET PREPARATION
# ============================================================

def prepare_dataset(
    clients=NUM_CLIENTS,
    test_size=TEST_SIZE,
    validation_size=VALIDATION_SIZE,
):

    set_seed()

    if not RAW.exists():

        raise FileNotFoundError(
            "\nPlantVillage dataset not found.\n\n"
            f"Place the ImageFolder dataset inside:\n"
            f"{RAW}\n"
        )

    print("\nPreparing TrustAgriLLM dataset...")

    if SPLITS.exists():

        print(
            "Removing previous dataset splits..."
        )

        shutil.rmtree(SPLITS)

    classes = sorted([
        path
        for path in RAW.iterdir()
        if path.is_dir()
    ])

    total_images = 0

    for cls in classes:

        images = sorted([
            image
            for image in cls.iterdir()
            if image.is_file()
        ])

        total_images += len(images)

        if len(images) < 5:

            print(
                f"Skipping {cls.name}: "
                "not enough images."
            )

            continue

        # -----------------------------------------
        # TRAIN / TEMP SPLIT
        # -----------------------------------------

        train_images, temp_images = train_test_split(

            images,

            test_size=(
                test_size + validation_size
            ),

            random_state=SEED,

            shuffle=True,
        )

        # -----------------------------------------
        # VALIDATION / TEST SPLIT
        # -----------------------------------------

        validation_fraction = (
            validation_size /
            (test_size + validation_size)
        )

        validation_images, test_images = (
            train_test_split(

                temp_images,

                test_size=(
                    1 - validation_fraction
                ),

                random_state=SEED,

                shuffle=True,
            )
        )

        # -----------------------------------------
        # COPY TEST
        # -----------------------------------------

        for image in test_images:

            destination = (
                SPLITS /
                "test" /
                cls.name /
                image.name
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copy2(
                image,
                destination,
            )

        # -----------------------------------------
        # DISTRIBUTE TRAIN DATA
        # -----------------------------------------

        buckets = [
            []
            for _ in range(clients)
        ]

        random.shuffle(train_images)

        for index, image in enumerate(
            train_images
        ):

            client_id = (
                index % clients
            )

            buckets[
                client_id
            ].append(image)

        for client_id, bucket in enumerate(
            buckets
        ):

            for image in bucket:

                destination = (
                    SPLITS /
                    f"client_{client_id}" /
                    "train" /
                    cls.name /
                    image.name
                )

                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                shutil.copy2(
                    image,
                    destination,
                )

        # -----------------------------------------
        # DISTRIBUTE VALIDATION DATA
        # -----------------------------------------

        validation_buckets = [
            []
            for _ in range(clients)
        ]

        random.shuffle(
            validation_images
        )

        for index, image in enumerate(
            validation_images
        ):

            client_id = (
                index % clients
            )

            validation_buckets[
                client_id
            ].append(image)

        for client_id, bucket in enumerate(
            validation_buckets
        ):

            for image in bucket:

                destination = (
                    SPLITS /
                    f"client_{client_id}" /
                    "validation" /
                    cls.name /
                    image.name
                )

                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                shutil.copy2(
                    image,
                    destination,
                )

    print("\nDataset preparation completed.")

    print(
        f"Total source images: "
        f"{total_images}"
    )

    print(
        f"Clients created: "
        f"{clients}"
    )

    print(
        "Train / Validation / Test "
        "separation completed."
    )


# ============================================================
# DATA LOADER
# ============================================================

def get_loader(
    path,
    train=False,
    max_samples=None,
):

    transform = (
        train_transform()
        if train
        else eval_transform()
    )

    dataset = datasets.ImageFolder(
        str(path),
        transform=transform,
    )

    # -----------------------------------------
    # LIMIT DATASET FOR CPU EXPERIMENTS
    # -----------------------------------------

    if (
        max_samples is not None
        and len(dataset) > max_samples
    ):

        generator = torch.Generator()

        generator.manual_seed(
            SEED
        )

        indices = torch.randperm(
            len(dataset),
            generator=generator,
        )[
            :max_samples
        ].tolist()

        dataset = Subset(
            dataset,
            indices,
        )

    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=train,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),
    )

    return dataset, loader


# ============================================================
# GET CLASSES
# ============================================================

def get_classes():

    dataset = datasets.ImageFolder(
        str(
            SPLITS /
            "test"
        )
    )

    return dataset.classes


# ============================================================
# DISEASE MODEL
# ============================================================

def create_model(
    num_classes,
):

    weights = (
        EfficientNet_B0_Weights.DEFAULT
        if USE_PRETRAINED
        else None
    )

    model = efficientnet_b0(
        weights=weights
    )

    input_features = (
        model.classifier[1].in_features
    )

    model.classifier[1] = nn.Linear(

        input_features,

        num_classes,
    )

    return model


# ============================================================
# TRAINING
# ============================================================

def train_model(

    model,

    loader,

    device,

    epochs=LOCAL_EPOCHS,

    client_id=None,

    round_number=None,
):

    model.to(device)

    model.train()

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LR,
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    start_time = (
        time.perf_counter()
    )

    epoch_losses = []

    for epoch in range(epochs):

        print(
            "\n"
            f"Client {client_id + 1} | "
            f"Round {round_number} | "
            f"Epoch {epoch + 1}/{epochs}",
            flush=True,
        )

        running_loss = 0.0

        total_batches = len(loader)

        if tqdm is not None:

            iterator = tqdm(

                loader,

                desc=(
                    f"Client "
                    f"{client_id + 1} "
                    f"Training"
                ),

                leave=False,
            )

        else:

            iterator = loader

        for batch_index, (
            images,
            labels,
        ) in enumerate(iterator):

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad()

            outputs = model(
                images
            )

            loss = criterion(

                outputs,

                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
            )

            if (
                tqdm is None
                and (
                    (
                        batch_index + 1
                    ) % 20 == 0
                    or (
                        batch_index
                        ==
                        total_batches - 1
                    )
                )
            ):

                print(

                    f"  Batch "
                    f"{batch_index + 1}/"
                    f"{total_batches} "
                    f"| Loss: "
                    f"{loss.item():.4f}",

                    flush=True,
                )

        average_loss = (
            running_loss /
            max(
                total_batches,
                1,
            )
        )

        epoch_losses.append(
            average_loss
        )

        print(

            f"Epoch completed "
            f"| Average Loss: "
            f"{average_loss:.4f}",

            flush=True,
        )

    training_time = (

        time.perf_counter()

        -

        start_time
    )

    return {

        "training_time": float(
            training_time
        ),

        "average_loss": float(

            np.mean(
                epoch_losses
            )
        ),
    }


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()

def evaluate(

    model,

    loader,

    device,
):

    model.to(device)

    model.eval()

    y_true = []

    y_pred = []

    for images, labels in loader:

        images = images.to(
            device
        )

        outputs = model(
            images
        )

        predictions = (
            outputs
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

        y_true.extend(
            labels.numpy()
        )

        y_pred.extend(
            predictions
        )

    precision, recall, f1, _ = (

        precision_recall_fscore_support(

            y_true,

            y_pred,

            average="weighted",

            zero_division=0,
        )
    )

    accuracy = accuracy_score(

        y_true,

        y_pred,
    )

    return {

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),
    }


# ============================================================
# FEDERATED LEARNING UTILITIES
# ============================================================

def clone_state(
    model,
):

    return {

        key: (
            value
            .detach()
            .cpu()
            .clone()
        )

        for key, value in (
            model
            .state_dict()
            .items()
        )
    }


def subtract_state(

    local_state,

    global_state,
):

    updates = {}

    for key in local_state:

        if (
            local_state[key]
            .dtype
            .is_floating_point
        ):

            updates[key] = (

                local_state[key]
                .float()

                -

                global_state[key]
                .float()
            )

    return updates


def flatten(
    update,
):

    tensors = [

        value.reshape(-1)

        for value in (
            update
            .values()
        )
    ]

    return torch.cat(
        tensors
    )


# ============================================================
# FEDAVG
# ============================================================

def weighted_fedavg(

    global_model,

    accepted_results,
):

    total_samples = sum(

        result["samples"]

        for result in (
            accepted_results
        )
    )

    keys = (

        accepted_results[0]
        ["state"]
        .keys()
    )

    new_state = {}

    for key in keys:

        reference = (

            accepted_results[0]
            ["state"][key]
        )

        if not (
            reference.dtype
            .is_floating_point
        ):

            new_state[key] = (
                reference
            )

            continue

        aggregated_value = sum(

            result["state"][key]
            .float()

            *

            (
                result["samples"]
                /
                total_samples
            )

            for result in (
                accepted_results
            )
        )

        new_state[key] = (

            aggregated_value
            .type(
                reference.dtype
            )
        )

    global_model.load_state_dict(
        new_state
    )

    return global_model


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(
    values,
):

    values = np.asarray(

        values,

        dtype=float,
    )

    if (

        len(values) == 1

        or

        np.isclose(
            values.max(),
            values.min(),
        )
    ):

        return np.ones(
            len(values)
        )

    return (

        values

        -

        values.min()

    ) / (

        values.max()

        -

        values.min()
    )


# ============================================================
# TRUST CALCULATION
# ============================================================

def compute_trust(
    results,
):

    # TRUST-AWARE UPDATE VALIDATION: evaluate the submitted update.
    quality = normalize([
        result["validation_accuracy"]
        for result in results
    ])
    participation = normalize([
        result["participation"]
        for result in results
    ])
    norms = np.asarray([
        result["update_norm"]
        for result in results
    ], dtype=float)

    median_norm = float(np.median(norms))
    mad = float(np.median(np.abs(norms - median_norm)))
    robust_z = (
        0.6745 * np.abs(norms - median_norm)
        / (mad + 1e-8)
    )
    relative_deviation = (
        np.abs(norms - median_norm)
        / (abs(median_norm) + 1e-8)
    )
    consistency = normalize(1.0 / (1.0 + relative_deviation))

    for index, result in enumerate(results):
        norm_ratio = (
            norms[index] / (abs(median_norm) + 1e-8)
        )
        update_anomaly = bool(
            robust_z[index] > 3.5
            or norm_ratio > 3.0
        )
        trust_score = (
            0.50 * quality[index]
            + 0.25 * participation[index]
            + 0.25 * consistency[index]
        )
        result["quality_score"] = float(quality[index])
        result["participation_score"] = float(participation[index])
        result["consistency_score"] = float(consistency[index])
        result["trust_score"] = float(trust_score)
        result["median_update_norm"] = float(median_norm)
        result["norm_ratio"] = float(norm_ratio)
        result["robust_z_score"] = float(robust_z[index])
        result["update_anomaly"] = update_anomaly
        result["rejection_reason"] = (
            "anomalous_update" if update_anomaly else ""
        )

    return results

# ============================================================
# BLOCKCHAIN LEDGER
# ============================================================

class PermissionedLedger:

    def __init__(
        self,
        filename,
    ):

        self.path = Path(
            filename
        )

        self.chain = []

        if self.path.exists():

            self.chain = (
                json.loads(
                    self.path.read_text()
                )
            )

        if not self.chain:

            self.append({

                "type":
                    "genesis",

                "message":
                    (
                        "TrustAgriLLM "
                        "provenance ledger"
                    ),
            })

    @staticmethod

    def sha256(
        obj,
    ):

        return hashlib.sha256(

            json.dumps(

                obj,

                sort_keys=True,

            ).encode()

        ).hexdigest()

    def append(
        self,
        payload,
    ):

        previous_hash = (

            self.chain[-1]["hash"]

            if self.chain

            else "0"
        )

        block = {

            "index":
                len(self.chain),

            "timestamp":
                time.time(),

            "previous_hash":
                previous_hash,

            "payload":
                payload,
        }

        block["hash"] = (
            self.sha256(
                block
            )
        )

        self.chain.append(
            block
        )

        self.path.write_text(

            json.dumps(

                self.chain,

                indent=2,
            )
        )

        return block

    def verify(
        self,
    ):

        for index, block in enumerate(
            self.chain
        ):

            expected_hash = (

                self.sha256({

                    "index":
                        block["index"],

                    "timestamp":
                        block[
                            "timestamp"
                        ],

                    "previous_hash":
                        block[
                            "previous_hash"
                        ],

                    "payload":
                        block[
                            "payload"
                        ],
                })
            )

            if (
                block["hash"]
                !=
                expected_hash
            ):

                return False

            if (
                index > 0
                and
                block[
                    "previous_hash"
                ]
                !=
                self.chain[
                    index - 1
                ]["hash"]
            ):

                return False

        return True


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(

    history,

    client_history,
):

    history_path = (

        RESULTS

        /

        "federated_history.csv"
    )

    client_path = (

        RESULTS

        /

        "client_trust_history.csv"
    )

    pd.DataFrame(
        history
    ).to_csv(

        history_path,

        index=False,
    )

    pd.DataFrame(
        client_history
    ).to_csv(

        client_path,

        index=False,
    )

    (

        RESULTS

        /

        "federated_history.json"

    ).write_text(

        json.dumps(

            history,

            indent=2,
        )
    )


# ============================================================
# FEDERATED EXPERIMENT
# ============================================================

def run_federated(
    malicious_clients=0,
):

    set_seed()

    device = get_device()

    print("\n")

    print("=" * 65)

    print(
        "TrustAgriLLM "
        "Federated Learning Started"
    )

    print("=" * 65)

    print(
        f"Device: "
        f"{device}"
    )

    print(
        f"Clients: "
        f"{NUM_CLIENTS}"
    )

    print(
        f"Communication Rounds: "
        f"{ROUNDS}"
    )

    print(
        f"Local Epochs: "
        f"{LOCAL_EPOCHS}"
    )

    print(
        f"Batch Size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Max Samples Per Client: "
        f"{MAX_SAMPLES_PER_CLIENT}"
    )

    print(
        f"Malicious Clients: "
        f"{malicious_clients}"
    )

    print("=" * 65)

    # -----------------------------------------
    # LOAD CLIENT DATA
    # -----------------------------------------

    client_data = []

    classes = None

    print(
        "\nLoading client datasets..."
    )

    for client_id in range(
        NUM_CLIENTS
    ):

        train_path = (

            SPLITS

            /

            f"client_{client_id}"

            /

            "train"
        )

        validation_path = (

            SPLITS

            /

            f"client_{client_id}"

            /

            "validation"
        )

        train_dataset, train_loader = (

            get_loader(

                train_path,

                train=True,

                max_samples=(
                    MAX_SAMPLES_PER_CLIENT
                ),
            )
        )

        validation_dataset, validation_loader = (

            get_loader(

                validation_path,

                train=False,

                max_samples=(
                    MAX_VALIDATION_SAMPLES
                ),
            )
        )

        client_data.append({

            "train_dataset":
                train_dataset,

            "train_loader":
                train_loader,

            "validation_dataset":
                validation_dataset,

            "validation_loader":
                validation_loader,
        })

        if classes is None:

            classes = (
                datasets.ImageFolder(

                    str(
                        train_path
                    )
                )
                .classes
            )

        print(

            f"Client "
            f"{client_id + 1}: "

            f"Train = "
            f"{len(train_dataset)}, "

            f"Validation = "
            f"{len(validation_dataset)}"
        )

    # -----------------------------------------
    # TEST DATA
    # -----------------------------------------

    print(
        "\nLoading independent "
        "test dataset..."
    )

    test_dataset, test_loader = (

        get_loader(

            SPLITS / "test",

            train=False,

            max_samples=(
                MAX_TEST_SAMPLES
            ),
        )
    )

    print(
        f"Independent Test Samples: "
        f"{len(test_dataset)}"
    )

    # -----------------------------------------
    # MODEL
    # -----------------------------------------

    global_model = create_model(
        len(classes)
    )

    # -----------------------------------------
    # LEDGER
    # -----------------------------------------

    ledger_path = (

        LEDGER

        /

        (
            "trustagrillm_ledger"
            ".json"
        )
    )

    # Start fresh ledger for experiment
    if ledger_path.exists():

        ledger_path.unlink()

    ledger = PermissionedLedger(
        ledger_path
    )

    # -----------------------------------------
    # HISTORIES
    # -----------------------------------------

    history = []

    client_history = []

    participation = np.ones(
        NUM_CLIENTS
    )

    # ========================================================
    # COMMUNICATION ROUNDS
    # ========================================================

    for round_number in range(

        1,

        ROUNDS + 1,
    ):

        round_start = (
            time.perf_counter()
        )

        print("\n")

        print("=" * 65)

        print(
            f"FEDERATED ROUND "
            f"{round_number}/{ROUNDS}"
        )

        print("=" * 65)

        global_state = clone_state(
            global_model
        )

        round_results = []

        # ====================================================
        # CLIENT TRAINING
        # ====================================================

        for client_id, client in enumerate(
            client_data
        ):

            print("\n")

            print(
                f"Starting Client "
                f"{client_id + 1}/"
                f"{NUM_CLIENTS}"
            )

            local_model = copy.deepcopy(
                global_model
            )

            training_result = train_model(

                local_model,

                client[
                    "train_loader"
                ],

                device,

                client_id=client_id,

                round_number=(
                    round_number
                ),
            )

            print(
                f"Client "
                f"{client_id + 1} "
                f"training completed "
                f"in "
                f"{training_result['training_time']:.2f} "
                f"seconds"
            )

            # -----------------------------------------
            # VALIDATION
            # -----------------------------------------

            validation_metrics = evaluate(

                local_model,

                client[
                    "validation_loader"
                ],

                device,
            )

            print(

                f"Validation Accuracy: "

                f"{validation_metrics['accuracy']:.4f}"

                f" | F1: "

                f"{validation_metrics['f1']:.4f}"
            )

            # -----------------------------------------
            # MODEL STATE
            # -----------------------------------------

            local_state = clone_state(
                local_model
            )

            # -----------------------------------------
            # MALICIOUS UPDATE SIMULATION
            # -----------------------------------------

            is_malicious = (

                client_id
                <
                malicious_clients
            )

            if is_malicious:

                print(
                    "WARNING: "
                    "Simulating malicious "
                    "model update."
                )

                for key, value in (
                    local_state.items()
                ):

                    if (
                        value.dtype
                        .is_floating_point
                    ):

                        local_state[key] = (

                            torch.randn_like(
                                value
                            )

                            * 5
                        )

            # -----------------------------------------
            # UPDATE NORM
            # -----------------------------------------

            update = subtract_state(

                local_state,

                global_state,
            )

            update_norm = float(

                torch.norm(

                    flatten(
                        update
                    )

                ).item()
            )

            # -----------------------------------------
            # HASH
            # -----------------------------------------

            update_bytes = (
                b"".join(

                    value
                    .cpu()
                    .numpy()
                    .tobytes()

                    for value in (
                        update
                        .values()
                    )
                )
            )

            model_hash = (
                hashlib.sha256(
                    update_bytes
                )
                .hexdigest()
            )

            # -----------------------------------------
            # STORE RESULT
            # -----------------------------------------

            result = {

                "client_id":
                    client_id,

                "state":
                    local_state,

                "samples":
                    len(
                        client[
                            "train_dataset"
                        ]
                    ),

                "validation_accuracy":
                    validation_metrics[
                        "accuracy"
                    ],

                "validation_precision":
                    validation_metrics[
                        "precision"
                    ],

                "validation_recall":
                    validation_metrics[
                        "recall"
                    ],

                "validation_f1":
                    validation_metrics[
                        "f1"
                    ],

                "participation":
                    float(
                        participation[
                            client_id
                        ]
                    ),

                "update_norm":
                    update_norm,

                "model_hash":
                    model_hash,

                "training_time":
                    training_result[
                        "training_time"
                    ],

                "training_loss":
                    training_result[
                        "average_loss"
                    ],

                "malicious":
                    is_malicious,
            }

            round_results.append(
                result
            )

        # ====================================================
        # TRUST CALCULATION
        # ====================================================

        print(
            "\nCalculating trust scores..."
        )

        round_results = compute_trust(
            round_results
        )

        # ====================================================
        # TRUST FILTERING
        # ====================================================

        accepted = [
            result
            for result in round_results
            if (
                result["trust_score"] >= TRUST_THRESHOLD
                and not result.get("update_anomaly", False)
            )
        ]

        print("\nTrust-aware update validation:")
        for result in round_results:
            status = (
                "REJECTED" if result.get("update_anomaly", False)
                else ("ACCEPTED" if result in accepted else "LOW_TRUST")
            )
            print(
                f"Client {result['client_id'] + 1} | "
                f"Trust: {result['trust_score']:.4f} | "
                f"Norm ratio: {result.get('norm_ratio', 0.0):.3f} | "
                f"Robust-Z: {result.get('robust_z_score', 0.0):.3f} | "
                f"{status}"
            )

        # Never fall back to an anomalous update.
        if not accepted:
            non_anomalous = [
                result for result in round_results
                if not result.get("update_anomaly", False)
            ]
            if not non_anomalous:
                raise RuntimeError(
                    "All client updates were rejected as anomalous. "
                    "Aggregation is stopped to protect the global model."
                )
            print("No non-anomalous client reached the trust threshold.")
            print("Selecting highest-trust non-anomalous client.")
            accepted = [
                max(
                    non_anomalous,
                    key=lambda result: result["trust_score"],
                )
            ]

        # ====================================================
        # BLOCKCHAIN VALIDATION
        # ====================================================

        print(
            "\nRecording provenance "
            "information..."
        )

        blockchain_start = (
            time.perf_counter()
        )

        accepted_ids = set([

            result[
                "client_id"
            ]

            for result in (
                accepted
            )
        ])

        for result in round_results:

            ledger.append({

                "round":
                    round_number,

                "node_id":
                    result[
                        "client_id"
                    ],

                "model_update_hash":
                    result[
                        "model_hash"
                    ],

                "trust_score":
                    result[
                        "trust_score"
                    ],

                "accepted":
                    (
                        result[
                            "client_id"
                        ]
                        in
                        accepted_ids
                    ),

                "malicious":
                    result[
                        "malicious"
                    ],

                "update_anomaly":
                    result.get(
                        "update_anomaly",
                        False,
                    ),

                "rejection_reason":
                    result.get(
                        "rejection_reason",
                        "",
                    ),
            })

        blockchain_latency = (

            time.perf_counter()

            -

            blockchain_start
        )

        ledger_integrity = (
            ledger.verify()
        )

        # ====================================================
        # FEDAVG
        # ====================================================

        print(
            "\nPerforming trust-aware "
            "FedAvg aggregation..."
        )

        weighted_fedavg(

            global_model,

            accepted,
        )

        # ====================================================
        # INDEPENDENT TEST EVALUATION
        # ====================================================

        print(
            "\nEvaluating global model "
            "on independent test set..."
        )

        test_metrics = evaluate(

            global_model,

            test_loader,

            device,
        )

        round_time = (

            time.perf_counter()

            -

            round_start
        )

        # ====================================================
        # ROUND RESULTS
        # ====================================================

        row = {

            "round":
                round_number,

            "accuracy":
                test_metrics[
                    "accuracy"
                ],

            "precision":
                test_metrics[
                    "precision"
                ],

            "recall":
                test_metrics[
                    "recall"
                ],

            "f1":
                test_metrics[
                    "f1"
                ],

            "trusted_clients":
                len(
                    accepted
                ),

            "total_clients":
                len(
                    round_results
                ),

            "mean_trust":
                float(

                    np.mean([

                        result[
                            "trust_score"
                        ]

                        for result in (
                            round_results
                        )
                    ])
                ),

            "mean_training_loss":
                float(

                    np.mean([

                        result[
                            "training_loss"
                        ]

                        for result in (
                            round_results
                        )
                    ])
                ),

            "mean_training_time_sec":
                float(

                    np.mean([

                        result[
                            "training_time"
                        ]

                        for result in (
                            round_results
                        )
                    ])
                ),

            "blockchain_validation_latency_sec":
                float(
                    blockchain_latency
                ),

            "ledger_integrity":
                bool(
                    ledger_integrity
                ),

            "round_time_sec":
                float(
                    round_time
                ),
        }

        history.append(
            row
        )

        # ====================================================
        # CLIENT HISTORY
        # ====================================================

        for result in round_results:

            client_history.append({

                "round":
                    round_number,

                "client_id":
                    result[
                        "client_id"
                    ],

                "validation_accuracy":
                    result[
                        "validation_accuracy"
                    ],

                "validation_precision":
                    result[
                        "validation_precision"
                    ],

                "validation_recall":
                    result[
                        "validation_recall"
                    ],

                "validation_f1":
                    result[
                        "validation_f1"
                    ],

                "quality_score":
                    result[
                        "quality_score"
                    ],

                "participation_score":
                    result[
                        "participation_score"
                    ],

                "consistency_score":
                    result[
                        "consistency_score"
                    ],

                "trust_score":
                    result[
                        "trust_score"
                    ],

                "update_norm":
                    result[
                        "update_norm"
                    ],

                "training_loss":
                    result[
                        "training_loss"
                    ],

                "training_time":
                    result[
                        "training_time"
                    ],

                "accepted":
                    (
                        result[
                            "client_id"
                        ]
                        in
                        accepted_ids
                    ),

                "malicious":
                    result[
                        "malicious"
                    ],
            })

        # ====================================================
        # SAVE AFTER EVERY ROUND
        # ====================================================

        save_results(

            history,

            client_history,
        )

        print("\n")

        print(
            "ROUND RESULT"
        )

        print("-" * 65)

        print(
            f"Accuracy: "
            f"{row['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{row['precision']:.4f}"
        )

        print(
            f"Recall: "
            f"{row['recall']:.4f}"
        )

        print(
            f"F1 Score: "
            f"{row['f1']:.4f}"
        )

        print(
            f"Trusted Clients: "
            f"{row['trusted_clients']}/"
            f"{row['total_clients']}"
        )

        print(
            f"Mean Trust Score: "
            f"{row['mean_trust']:.4f}"
        )

        print(
            f"Blockchain Latency: "
            f"{row['blockchain_validation_latency_sec']:.6f} sec"
        )

        print(
            f"Ledger Integrity: "
            f"{row['ledger_integrity']}"
        )

        print(
            f"Round Time: "
            f"{row['round_time_sec']:.2f} sec"
        )

        print("-" * 65)

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    torch.save({

        "state_dict":
            global_model.state_dict(),

        "classes":
            classes,

    }, MODELS / "federated_global.pt")

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n")

    print("=" * 65)

    print(
        "FEDERATED LEARNING "
        "COMPLETED SUCCESSFULLY"
    )

    print("=" * 65)

    print(
        pd.DataFrame(
            history
        )
    )

    print("\nResults saved to:")

    print(
        RESULTS /
        "federated_history.csv"
    )

    print(
        RESULTS /
        "client_trust_history.csv"
    )

    print(
        MODELS /
        "federated_global.pt"
    )

    return history


# ============================================================
# MULTIMODAL CONTEXT
# ============================================================

def build_context(

    image_path,

    prediction,

    confidence,

    sensor,

    weather,

    farmer_note,

    crop,
):

    return {

        "image_path":
            str(
                image_path
            ),

        "disease_prediction":
            prediction,

        "confidence":
            float(
                confidence
            ),

        "sensor_data":
            sensor,

        "weather":
            weather,

        "farmer_observation":
            farmer_note,

        "crop_information":
            crop,
    }


# ============================================================
# MLLM PROMPT
# ============================================================

def context_prompt(
    context,
):

    return f"""
You are an agricultural decision-support assistant.

Use the plant image and all structured context below.

Do not invent measurements.

Clearly distinguish direct observations from inferences.

Disease prediction:
{context['disease_prediction']}

Prediction confidence:
{context['confidence']}

Sensor data:
{context['sensor_data']}

Weather information:
{context['weather']}

Farmer observation:
{context['farmer_observation']}

Crop information:
{context['crop_information']}

Return the following sections:

1. Disease interpretation
2. Contextual explanation
3. Contextual severity interpretation
4. Environmental risk assessment
5. Agricultural decision-support guidance

Avoid unsupported chemical dosage and
region-specific regulatory claims.
"""


# ============================================================
# MLLM
# ============================================================

def run_mllm(
    context,
):

    try:

        from transformers import (

            AutoProcessor,

            Qwen2_5_VLForConditionalGeneration,
        )

    except ImportError:

        raise RuntimeError(

            "\nInstall required packages:\n"

            "pip install transformers "
            "accelerate qwen-vl-utils\n"
        )

    model_id = (
        "Qwen/Qwen2.5-VL-3B-Instruct"
    )

    print(
        "\nLoading Qwen2.5-VL..."
    )

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(

            model_id,

            torch_dtype="auto",

            device_map="auto",
        )
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            model_id
        )
    )

    messages = [{

        "role":
            "user",

        "content": [

            {

                "type":
                    "image",

                "image":
                    (
                        "file://"
                        +
                        context[
                            "image_path"
                        ]
                    ),
            },

            {

                "type":
                    "text",

                "text":
                    context_prompt(
                        context
                    ),
            },
        ],
    }]

    text = (
        processor
        .apply_chat_template(

            messages,

            tokenize=False,

            add_generation_prompt=True,
        )
    )

    image = (
        Image
        .open(
            context[
                "image_path"
            ]
        )
        .convert("RGB")
    )

    inputs = (
        processor(

            text=[text],

            images=[image],

            padding=True,

            return_tensors="pt",
        )
        .to(
            model.device
        )
    )

    print(
        "Generating agricultural "
        "decision support..."
    )

    generated = (
        model.generate(

            **inputs,

            max_new_tokens=350,
        )
    )

    trimmed = (

        generated[
            :,
            inputs.input_ids.shape[1]:
        ]
    )

    output = (

        processor
        .batch_decode(

            trimmed,

            skip_special_tokens=True,

            clean_up_tokenization_spaces=False,
        )[0]
    )

    return output


# ============================================================
# MLLM DEMO
# ============================================================

def demo_mllm(

    image_path,

    prediction,

    confidence,
):

    context = build_context(

        image_path=image_path,

        prediction=prediction,

        confidence=confidence,

        sensor={

            "temperature_c":
                29,

            "humidity_percent":
                85,

            "soil_moisture_percent":
                70,

            "soil_ph":
                6.5,
        },

        weather={

            "rainfall":
                "moderate",

            "condition":
                "humid",
        },

        farmer_note=(

            "Brown lesions are "
            "spreading to neighboring leaves."
        ),

        crop={

            "crop":
                "Tomato",

            "growth_stage":
                "vegetative",
        },
    )

    output = run_mllm(
        context
    )

    result = {

        "context":
            context,

        "mllm_output":
            output,
    }

    output_path = (

        RESULTS

        /

        "mllm_output.json"
    )

    output_path.write_text(

        json.dumps(

            result,

            indent=2,
        )
    )

    print("\n")

    print("=" * 65)

    print(
        "MLLM AGRICULTURAL "
        "INTELLIGENCE OUTPUT"
    )

    print("=" * 65)

    print(
        output
    )

    print("\nSaved to:")

    print(
        output_path
    )


# ============================================================
# SUMMARY
# ============================================================

def show_summary():

    history_path = (

        RESULTS

        /

        "federated_history.csv"
    )

    trust_path = (

        RESULTS

        /

        "client_trust_history.csv"
    )

    if not (
        history_path.exists()
    ):

        raise FileNotFoundError(

            "\nRun federated experiment first.\n"
        )

    history = pd.read_csv(
        history_path
    )

    print("\n")

    print("=" * 65)

    print(
        "TrustAgriLLM "
        "EXPERIMENT SUMMARY"
    )

    print("=" * 65)

    print("\nFederated Round Results:")

    print(
        history.tail()
    )

    if trust_path.exists():

        trust_data = pd.read_csv(
            trust_path
        )

        print(
            "\nAverage Trust Score:"
        )

        print(

            trust_data[
                "trust_score"
            ]
            .mean()
        )

        print(
            "\nAccepted Updates:"
        )

        print(

            trust_data[
                "accepted"
            ]
            .sum()
        )

        print(
            "\nRejected Updates:"
        )

        print(

            len(
                trust_data
            )

            -

            trust_data[
                "accepted"
            ]
            .sum()
        )


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "TrustAgriLLM "
            "Complete Prototype"
        )
    )

    parser.add_argument(

        "command",

        choices=[

            "prepare",

            "federated",

            "mllm",

            "summary",
        ],
    )

    parser.add_argument(

        "--malicious_clients",

        type=int,

        default=0,
    )

    parser.add_argument(

        "--image",

        type=str,
    )

    parser.add_argument(

        "--prediction",

        default=(
            "Tomato Early Blight"
        ),
    )

    parser.add_argument(

        "--confidence",

        type=float,

        default=0.92,
    )

    args = parser.parse_args()

    # ========================================================
    # PREPARE
    # ========================================================

    if (
        args.command
        ==
        "prepare"
    ):

        prepare_dataset()

    # ========================================================
    # FEDERATED
    # ========================================================

    elif (
        args.command
        ==
        "federated"
    ):

        if not SPLITS.exists():

            raise FileNotFoundError(

                "\nDataset splits not found.\n"

                "Run:\n"

                "python trustagrillm.py prepare\n"
            )

        run_federated(

            malicious_clients=(
                args
                .malicious_clients
            )
        )

    # ========================================================
    # MLLM
    # ========================================================

    elif (
        args.command
        ==
        "mllm"
    ):

        if not args.image:

            raise ValueError(

                "\n--image is required.\n\n"

                "Example:\n"

                'python trustagrillm.py mllm '
                '--image "leaf.jpg" '
                '--prediction "Tomato Early Blight" '
                '--confidence 0.92\n'
            )

        demo_mllm(

            args.image,

            args.prediction,

            args.confidence,
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    elif (
        args.command
        ==
        "summary"
    ):

        show_summary()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    main()