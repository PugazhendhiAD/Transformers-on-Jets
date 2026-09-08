import os
import glob
import math
import random

import numpy as np
import awkward as ak
import uproot

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Configuration
# ============================================================

DATA_DIR = r"C:\Users\pugaz\Downloads\JetClass_Pythia_train_100M_part0"

MAX_PARTICLES = 128

BATCH_SIZE = 32

D_MODEL = 64

N_HEADS = 4

N_PARTICLE_BLOCKS = 2

N_CLASS_BLOCKS = 1

DROPOUT = 0.1

EPOCHS = 20

LEARNING_RATE = 1e-3

MAX_EVENTS = 100000


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)


# ============================================================
# 2. JetClass labels
# ============================================================

LABEL_NAMES = [
    "QCD",
    "Hbb",
    "Hcc",
    "Hgg",
    "H4q",
    "Hqql",
    "Zqq",
    "Wqq",
    "Tbqq",
    "Tbl"
]


LABEL_BRANCHES = [
    "label_QCD",
    "label_Hbb",
    "label_Hcc",
    "label_Hgg",
    "label_H4q",
    "label_Hqql",
    "label_Zqq",
    "label_Wqq",
    "label_Tbqq",
    "label_Tbl"
]


# ============================================================
# 3. Select 2 ROOT files from each class
# ============================================================

import glob
import os
import random
import numpy as np
import awkward as ak
import uproot


# Reproducible random selection
random.seed(42)


# ------------------------------------------------------------
# Filename prefix for each JetClass class
# ------------------------------------------------------------

CLASS_PREFIXES = {
    "QCD": "ZJetsToNuNu",
    "Hbb": "HToBB",
    "Hcc": "HToCC",
    "Hgg": "HToGG",
    "H4q": "HToWW4Q",
    "Hqql": "HToWW2Q1L",
    "Zqq": "ZToQQ",
    "Wqq": "WToQQ",
    "Tbqq": "TTBar",
    "Tbl": "TTBarLep"
}


# ------------------------------------------------------------
# Number of files per class
# ------------------------------------------------------------

FILES_PER_CLASS = 2


# ------------------------------------------------------------
# Number of jets to use from EACH class
# ------------------------------------------------------------

MAX_EVENTS_PER_CLASS = 5000


selected_files = {}


# ------------------------------------------------------------
# Find files for every class
# ------------------------------------------------------------

for class_name, prefix in CLASS_PREFIXES.items():

    class_files = sorted(
        glob.glob(
            os.path.join(
                DATA_DIR,
                prefix + "_*.root"
            )
        )
    )


    print(
        f"\n{class_name}: found {len(class_files)} files"
    )


    if len(class_files) < FILES_PER_CLASS:

        raise RuntimeError(
            f"Could not find "
            f"{FILES_PER_CLASS} files for {class_name}"
        )


    chosen = random.sample(
        class_files,
        FILES_PER_CLASS
    )


    selected_files[class_name] = chosen


    print("Selected:")

    for filename in chosen:

        print(
            "   ",
            os.path.basename(filename)
        )


print(
    "\n========================================"
)

print(
    "Total selected ROOT files:",
    sum(
        len(files)
        for files in selected_files.values()
    )
)

print(
    "========================================"
)

# ============================================================
# 4. Load balanced JetClass data
# ============================================================

def load_class_data(
    class_name,
    filenames,
    max_events
):

    print(
        f"\nLoading class: {class_name}"
    )


    class_px = []
    class_py = []
    class_pz = []
    class_energy = []

    class_deta = []
    class_dphi = []


    events_loaded = 0


    # --------------------------------------------------------
    # Branches we need
    # --------------------------------------------------------

    branches = [
        "part_px",
        "part_py",
        "part_pz",
        "part_energy",
        "part_deta",
        "part_dphi"
    ]


    # --------------------------------------------------------
    # Read the selected files for this class
    # --------------------------------------------------------

    for filename in filenames:

        if events_loaded >= max_events:

            break


        print(
            "Reading:",
            os.path.basename(filename)
        )


        with uproot.open(filename) as f:

            tree = f["tree"]


            # How many events remain to be loaded?

            remaining = (
                max_events
                - events_loaded
            )


            # Read only the required number of events
            arrays = tree.arrays(
                branches,
                entry_start=0,
                entry_stop=remaining,
                library="ak"
            )


        n_loaded = len(
            arrays["part_px"]
        )


        # ----------------------------------------------------
        # Store particle information
        # ----------------------------------------------------

        class_px.append(
            arrays["part_px"]
        )

        class_py.append(
            arrays["part_py"]
        )

        class_pz.append(
            arrays["part_pz"]
        )

        class_energy.append(
            arrays["part_energy"]
        )

        class_deta.append(
            arrays["part_deta"]
        )

        class_dphi.append(
            arrays["part_dphi"]
        )


        events_loaded += n_loaded


        print(
            f"Loaded from file: {n_loaded}"
        )

        print(
            f"Total class events: "
            f"{events_loaded}"
        )


    # --------------------------------------------------------
    # Combine files belonging to this class
    # --------------------------------------------------------

    px = ak.concatenate(
        class_px
    )

    py = ak.concatenate(
        class_py
    )

    pz = ak.concatenate(
        class_pz
    )

    energy = ak.concatenate(
        class_energy
    )

    deta = ak.concatenate(
        class_deta
    )

    dphi = ak.concatenate(
        class_dphi
    )


    # --------------------------------------------------------
    # Create class labels
    #
    # The order follows LABEL_NAMES:
    #
    # 0 → QCD
    # 1 → Hbb
    # 2 → Hcc
    # ...
    # --------------------------------------------------------

    class_index = LABEL_NAMES.index(
        class_name
    )


    labels = np.full(
        events_loaded,
        class_index,
        dtype=np.int64
    )


    return (
        px,
        py,
        pz,
        energy,
        deta,
        dphi,
        labels
    )

# ============================================================
# 5. Load all classes
# ============================================================

all_px = []
all_py = []
all_pz = []
all_energy = []

all_deta = []
all_dphi = []

all_labels = []


for class_name in LABEL_NAMES:

    (
        px,
        py,
        pz,
        energy,
        deta,
        dphi,
        labels
    ) = load_class_data(
        class_name,
        selected_files[class_name],
        MAX_EVENTS_PER_CLASS
    )


    all_px.append(px)
    all_py.append(py)
    all_pz.append(pz)
    all_energy.append(energy)

    all_deta.append(deta)
    all_dphi.append(dphi)

    all_labels.append(labels)


# ============================================================
# Combine all classes
# ============================================================

px = ak.concatenate(
    all_px
)

py = ak.concatenate(
    all_py
)

pz = ak.concatenate(
    all_pz
)

energy = ak.concatenate(
    all_energy
)

deta = ak.concatenate(
    all_deta
)

dphi = ak.concatenate(
    all_dphi
)

labels = np.concatenate(
    all_labels
)


# ============================================================
# Check dataset
# ============================================================

print(
    "\n========================================"
)

print(
    "Total events:",
    len(labels)
)

print(
    "========================================"
)


print(
    "\nEvents per class:"
)


for i, name in enumerate(
    LABEL_NAMES
):

    count = np.sum(
        labels == i
    )

    print(
        f"{i}: {name:6s} → {count}"
    )

# ============================================================
# 6. Convert particle information
# ============================================================

def prepare_event(
    px,
    py,
    pz,
    energy,
    deta,
    dphi,
    max_particles
):

    # --------------------------------------------------------
    # Transverse momentum
    # --------------------------------------------------------

    pt = np.sqrt(
        px**2 + py**2
    )


    # --------------------------------------------------------
    # Logarithmic features
    #
    # Particle energies and momenta span a large range.
    # log(1+x) keeps the numbers manageable.
    # --------------------------------------------------------

    log_pt = np.log1p(
        pt
    )

    log_energy = np.log1p(
        energy
    )


    # --------------------------------------------------------
    # Keep at most max_particles
    # --------------------------------------------------------

    n = min(
        len(pt),
        max_particles
    )


    features = np.stack(
        [
            log_pt[:n],
            deta[:n],
            dphi[:n],
            log_energy[:n]
        ],
        axis=1
    )


    return features


# ============================================================
# 7. Dataset
# ============================================================

class JetClassDataset(Dataset):

    def __init__(
        self,
        px,
        py,
        pz,
        energy,
        deta,
        dphi,
        labels,
        max_particles
    ):

        self.events = []

        self.labels = labels


        for i in range(
            len(labels)
        ):

            event = prepare_event(
                px[i],
                py[i],
                pz[i],
                energy[i],
                deta[i],
                dphi[i],
                max_particles
            )


            self.events.append(
                torch.tensor(
                    event,
                    dtype=torch.float32
                )
            )


    def __len__(self):

        return len(
            self.events
        )


    def __getitem__(
        self,
        index
    ):

        return (
            self.events[index],
            torch.tensor(
                self.labels[index],
                dtype=torch.long
            )
        )


# ============================================================
# 8. Train/test split
# ============================================================

indices = np.random.permutation(
    len(labels)
)


split = int(
    0.8 * len(indices)
)


train_indices = indices[
    :split
]

test_indices = indices[
    split:
]


def subset_awkward(
    array,
    indices
):

    return array[
        indices
    ]


train_dataset = JetClassDataset(
    subset_awkward(px, train_indices),
    subset_awkward(py, train_indices),
    subset_awkward(pz, train_indices),
    subset_awkward(energy, train_indices),
    subset_awkward(deta, train_indices),
    subset_awkward(dphi, train_indices),
    labels[train_indices],
    MAX_PARTICLES
)


test_dataset = JetClassDataset(
    subset_awkward(px, test_indices),
    subset_awkward(py, test_indices),
    subset_awkward(pz, test_indices),
    subset_awkward(energy, test_indices),
    subset_awkward(deta, test_indices),
    subset_awkward(dphi, test_indices),
    labels[test_indices],
    MAX_PARTICLES
)


print(
    "Training events:",
    len(train_dataset)
)

print(
    "Testing events:",
    len(test_dataset)
)


# ============================================================
# 9. Collate function
# ============================================================

def collate_fn(batch):

    events, labels = zip(*batch)


    max_particles = max(
        event.shape[0]
        for event in events
    )


    batch_size = len(events)


    padded = torch.zeros(
        batch_size,
        max_particles,
        4
    )


    padding_mask = torch.ones(
        batch_size,
        max_particles,
        dtype=torch.bool
    )


    for i, event in enumerate(events):

        n = event.shape[0]


        padded[
            i,
            :n
        ] = event


        padding_mask[
            i,
            :n
        ] = False


    labels = torch.stack(
        labels
    )


    return (
        padded,
        padding_mask,
        labels
    )


# ============================================================
# 10. DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


# ============================================================
# 11. Pairwise interaction embedding
# ============================================================

class PairwiseEmbedding(nn.Module):

    def __init__(
        self,
        n_heads
    ):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                3,
                32
            ),

            nn.GELU(),

            nn.Linear(
                32,
                n_heads
            )
        )


    def forward(
        self,
        x
    ):

        # x contains:
        #
        # deta
        # dphi
        # log(pt)
        # energy
        #
        # We construct pairwise quantities
        # from the first three useful coordinates.


        deta = (
            x[:, :, None, 1]
            -
            x[:, None, :, 1]
        )


        dphi = (
            x[:, :, None, 2]
            -
            x[:, None, :, 2]
        )


        # Wrap dphi to [-pi, pi]

        dphi = (
            torch.atan2(
                torch.sin(dphi),
                torch.cos(dphi)
            )
        )


        dR = torch.sqrt(
            deta**2
            +
            dphi**2
            +
            1e-8
        )


        pairwise = torch.stack(
            [
                deta,
                dphi,
                dR
            ],
            dim=-1
        )


        bias = self.network(
            pairwise
        )


        # ----------------------------------------------------
        # Shape:
        #
        # batch × heads × particles × particles
        # ----------------------------------------------------

        bias = bias.permute(
            0,
            3,
            1,
            2
        )


        return bias


# ============================================================
# 12. Particle Multi Head Attention
# ============================================================

class ParticleAttention(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        dropout
    ):

        super().__init__()


        assert (
            d_model % n_heads == 0
        )


        self.d_model = d_model

        self.n_heads = n_heads

        self.head_dim = (
            d_model // n_heads
        )


        self.q = nn.Linear(
            d_model,
            d_model
        )

        self.k = nn.Linear(
            d_model,
            d_model
        )

        self.v = nn.Linear(
            d_model,
            d_model
        )

        self.out = nn.Linear(
            d_model,
            d_model
        )


        self.dropout = nn.Dropout(
            dropout
        )


    def forward(
        self,
        x,
        pairwise_bias,
        padding_mask
    ):

        B, N, D = x.shape


        Q = self.q(x)

        K = self.k(x)

        V = self.v(x)


        # Split into heads

        Q = Q.view(
            B,
            N,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)


        K = K.view(
            B,
            N,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)


        V = V.view(
            B,
            N,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)


        # ----------------------------------------------------
        # QK^T
        # ----------------------------------------------------

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )


        # ----------------------------------------------------
        # Scale
        # ----------------------------------------------------

        scores = scores / math.sqrt(
            self.head_dim
        )


        # ----------------------------------------------------
        # Add Particle Transformer pairwise bias
        # ----------------------------------------------------

        scores = (
            scores
            +
            pairwise_bias
        )


        # ----------------------------------------------------
        # Mask padding particles
        # ----------------------------------------------------

        mask = padding_mask[
            :, None, None, :
        ]


        scores = scores.masked_fill(
            mask,
            -1e9
        )


        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        attention = torch.softmax(
            scores,
            dim=-1
        )


        attention = self.dropout(
            attention
        )


        # ----------------------------------------------------
        # Weighted V
        # ----------------------------------------------------

        output = torch.matmul(
            attention,
            V
        )


        # Combine heads

        output = output.transpose(
            1,
            2
        ).contiguous()


        output = output.view(
            B,
            N,
            D
        )


        return self.out(
            output
        )


# ============================================================
# 13. Particle Attention Block
# ============================================================

class ParticleAttentionBlock(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        dropout
    ):

        super().__init__()


        self.norm1 = nn.LayerNorm(
            d_model
        )


        self.attention = ParticleAttention(
            d_model,
            n_heads,
            dropout
        )


        self.norm2 = nn.LayerNorm(
            d_model
        )


        self.ffn = nn.Sequential(

            nn.Linear(
                d_model,
                4 * d_model
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                4 * d_model,
                d_model
            )
        )


        self.dropout = nn.Dropout(
            dropout
        )


    def forward(
        self,
        x,
        pairwise_bias,
        padding_mask
    ):


        # ----------------------------------------------------
        # Particle Attention
        # ----------------------------------------------------

        y = self.norm1(
            x
        )


        y = self.attention(
            y,
            pairwise_bias,
            padding_mask
        )


        x = x + self.dropout(
            y
        )


        # ----------------------------------------------------
        # Feed Forward
        # ----------------------------------------------------

        y = self.norm2(
            x
        )


        y = self.ffn(
            y
        )


        x = x + self.dropout(
            y
        )


        return x


# ============================================================
# 14. Class Attention
# ============================================================

class ClassAttention(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        dropout
    ):

        super().__init__()


        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )


        self.norm1 = nn.LayerNorm(
            d_model
        )


        self.norm2 = nn.LayerNorm(
            d_model
        )


        self.ffn = nn.Sequential(

            nn.Linear(
                d_model,
                4 * d_model
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                4 * d_model,
                d_model
            )
        )


        self.dropout = nn.Dropout(
            dropout
        )


    def forward(
        self,
        class_token,
        particles,
        padding_mask
    ):


        # ----------------------------------------------------
        # Class token queries particle representations
        # ----------------------------------------------------

        q = self.norm1(
            class_token
        )

        kv = self.norm1(
            particles
        )


        attended, _ = self.attention(
            q,
            kv,
            kv,
            key_padding_mask=padding_mask
        )


        class_token = (
            class_token
            +
            self.dropout(
                attended
            )
        )


        # ----------------------------------------------------
        # Feed Forward
        # ----------------------------------------------------

        y = self.norm2(
            class_token
        )


        y = self.ffn(
            y
        )


        class_token = (
            class_token
            +
            self.dropout(
                y
            )
        )


        return class_token


# ============================================================
# 15. Complete simplified Particle Transformer
# ============================================================

class ParticleTransformer(nn.Module):

    def __init__(
        self,
        input_dim=4,
        d_model=64,
        n_heads=4,
        n_particle_blocks=2,
        n_class_blocks=1,
        dropout=0.1,
        num_classes=10
    ):

        super().__init__()


        # ----------------------------------------------------
        # Particle embedding
        # ----------------------------------------------------

        self.embedding = nn.Sequential(

            nn.Linear(
                input_dim,
                d_model
            ),

            nn.GELU(),

            nn.Linear(
                d_model,
                d_model
            )
        )


        # ----------------------------------------------------
        # Pairwise interaction embedding
        # ----------------------------------------------------

        self.pairwise_embedding = PairwiseEmbedding(
            n_heads
        )


        # ----------------------------------------------------
        # Particle Attention Blocks
        # ----------------------------------------------------

        self.particle_blocks = nn.ModuleList([

            ParticleAttentionBlock(
                d_model,
                n_heads,
                dropout
            )

            for _ in range(
                n_particle_blocks
            )

        ])


        # ----------------------------------------------------
        # Learnable class token
        # ----------------------------------------------------

        self.class_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                d_model
            )
        )


        # ----------------------------------------------------
        # Class Attention Blocks
        # ----------------------------------------------------

        self.class_blocks = nn.ModuleList([

            ClassAttention(
                d_model,
                n_heads,
                dropout
            )

            for _ in range(
                n_class_blocks
            )

        ])


        # ----------------------------------------------------
        # Final classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.LayerNorm(
                d_model
            ),

            nn.Linear(
                d_model,
                d_model
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                d_model,
                num_classes
            )
        )


    def forward(
        self,
        particles,
        padding_mask
    ):


        # ----------------------------------------------------
        # Particle embedding
        # ----------------------------------------------------

        x = self.embedding(
            particles
        )


        # ----------------------------------------------------
        # Pairwise interaction information
        # ----------------------------------------------------

        pairwise_bias = (
            self.pairwise_embedding(
                particles
            )
        )


        # ----------------------------------------------------
        # Particle Attention Blocks
        # ----------------------------------------------------

        for block in self.particle_blocks:

            x = block(
                x,
                pairwise_bias,
                padding_mask
            )


        # ----------------------------------------------------
        # Create class token
        # ----------------------------------------------------

        B = x.size(0)


        class_token = (
            self.class_token
            .expand(
                B,
                -1,
                -1
            )
        )


        # ----------------------------------------------------
        # Class Attention Blocks
        # ----------------------------------------------------

        for block in self.class_blocks:

            class_token = block(
                class_token,
                x,
                padding_mask
            )


        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        class_representation = (
            class_token[:, 0, :]
        )


        output = self.classifier(
            class_representation
        )


        return output


# ============================================================
# 16. Create model
# ============================================================

model = ParticleTransformer(
    input_dim=4,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_particle_blocks=N_PARTICLE_BLOCKS,
    n_class_blocks=N_CLASS_BLOCKS,
    dropout=DROPOUT,
    num_classes=10
).to(device)


print("\nModel created.")


# ============================================================
# 17. Loss and optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# 18. Training
# ============================================================

for epoch in range(
    EPOCHS
):

    model.train()


    total_loss = 0

    correct = 0

    total = 0


    for (
        particles,
        padding_mask,
        labels
    ) in train_loader:


        particles = particles.to(
            device
        )

        padding_mask = padding_mask.to(
            device
        )

        labels = labels.to(
            device
        )


        # Forward

        logits = model(
            particles,
            padding_mask
        )


        # Loss

        loss = criterion(
            logits,
            labels
        )


        # Backpropagation

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        total_loss += loss.item()


        # Accuracy

        predictions = logits.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


    train_loss = (
        total_loss
        / len(train_loader)
    )


    train_accuracy = (
        correct
        / total
    )


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"Loss = {train_loss:.4f} "
        f"Accuracy = {train_accuracy:.4f}"
    )


# ============================================================
# 19. Test
# ============================================================

model.eval()

correct = 0

total = 0


with torch.no_grad():

    for (
        particles,
        padding_mask,
        labels
    ) in test_loader:


        particles = particles.to(
            device
        )

        padding_mask = padding_mask.to(
            device
        )

        labels = labels.to(
            device
        )


        logits = model(
            particles,
            padding_mask
        )


        predictions = logits.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


test_accuracy = (
    correct / total
)


print(
    "\nTest accuracy:",
    test_accuracy
)