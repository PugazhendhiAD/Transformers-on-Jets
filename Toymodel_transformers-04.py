import math
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)


# ============================================================
# 2. Reproducibility
# ============================================================

torch.manual_seed(42)
random.seed(42)


# ============================================================
# 3. Dataset
# ============================================================

class ParticleDataset(Dataset):

    def __init__(
        self,
        n_events=5000,
        min_particles=20,
        max_particles=80
    ):

        self.events = []

        self.labels = []

        for _ in range(n_events):

            # Random number of particles
            n_particles = random.randint(
                min_particles,
                max_particles
            )


            # Random class
            label = random.randint(0, 1)


            # ------------------------------------------------
            # Class 0:
            # broadly distributed particles
            # ------------------------------------------------

            if label == 0:

                pt = torch.rand(
                    n_particles
                ) * 5.0

                eta = (
                    torch.rand(n_particles) * 6.0
                    - 3.0
                )

                phi = (
                    torch.rand(n_particles)
                    * 2 * math.pi
                    - math.pi
                )


            # ------------------------------------------------
            # Class 1:
            # clustered particles
            # ------------------------------------------------

            else:

                pt = torch.rand(
                    n_particles
                ) * 5.0

                # Choose a cluster center
                eta_center = (
                    torch.rand(1) * 2.0 - 1.0
                )

                phi_center = (
                    torch.rand(1) * 2 * math.pi
                    - math.pi
                )


                # Particles concentrated around center
                eta = (
                    eta_center
                    + 0.3 * torch.randn(
                        n_particles
                    )
                )

                phi = (
                    phi_center
                    + 0.3 * torch.randn(
                        n_particles
                    )
                )


                # Keep phi within [-pi, pi]

                phi = (
                    (phi + math.pi)
                    % (2 * math.pi)
                    - math.pi
                )


            # ------------------------------------------------
            # Combine particle features
            # ------------------------------------------------

            particles = torch.stack(
                [
                    pt,
                    eta,
                    phi
                ],
                dim=1
            )


            self.events.append(
                particles
            )

            self.labels.append(
                label
            )


    def __len__(self):

        return len(self.events)


    def __getitem__(self, index):

        return (
            self.events[index],
            torch.tensor(
                self.labels[index],
                dtype=torch.long
            )
        )


# ============================================================
# 4. Create dataset
# ============================================================

dataset = ParticleDataset(
    n_events=5000
)


print(
    "Number of events:",
    len(dataset)
)


# Look at one event

event, label = dataset[0]

print(
    "Particles in event:",
    event.shape
)

print(
    "Label:",
    label
)

print(
    "First 5 particles:\n",
    event[:5]
)


# ============================================================
# 5. Train / test split
# ============================================================

train_size = int(
    0.8 * len(dataset)
)

test_size = (
    len(dataset)
    - train_size
)


train_dataset, test_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, test_size]
)


# ============================================================
# 6. Collate function
# ============================================================

def collate_fn(batch):

    events, labels = zip(*batch)


    # Number of particles differs between events
    #
    # Therefore we need padding.

    max_particles = max(
        event.shape[0]
        for event in events
    )


    batch_size = len(events)

    n_features = events[0].shape[1]


    # --------------------------------------------------------
    # Create padded tensor
    # --------------------------------------------------------

    padded_events = torch.zeros(
        batch_size,
        max_particles,
        n_features
    )


    # --------------------------------------------------------
    # Mask
    #
    # True = padding
    # False = real particle
    # --------------------------------------------------------

    padding_mask = torch.ones(
        batch_size,
        max_particles,
        dtype=torch.bool
    )


    # --------------------------------------------------------
    # Insert events
    # --------------------------------------------------------

    for i, event in enumerate(events):

        n_particles = event.shape[0]


        padded_events[
            i,
            :n_particles,
            :
        ] = event


        padding_mask[
            i,
            :n_particles
        ] = False


    labels = torch.stack(
        labels
    )


    return (
        padded_events,
        padding_mask,
        labels
    )


# ============================================================
# 7. DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn
)


test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn
)


# ============================================================
# 8. Particle Transformer
# ============================================================

class ParticleTransformer(nn.Module):

    def __init__(
        self,
        input_dim=3,
        d_model=64,
        n_heads=4,
        n_layers=3,
        dropout=0.1
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
        # Transformer encoder
        # ----------------------------------------------------

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True
            )
        )


        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_layers
            )
        )


        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

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
                2
            )
        )


    def forward(
        self,
        particles,
        padding_mask
    ):

        # ----------------------------------------------------
        # particles:
        #
        # (batch, particles, features)
        #
        # Example:
        #
        # (64, 80, 3)
        # ----------------------------------------------------


        # ----------------------------------------------------
        # Embed each particle
        # ----------------------------------------------------

        x = self.embedding(
            particles
        )


        # ----------------------------------------------------
        # Transformer
        #
        # padding_mask tells the Transformer which
        # entries are fake padding.
        # ----------------------------------------------------

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )


        # ----------------------------------------------------
        # Convert particle representations into an
        # event representation
        #
        # We use masked mean pooling.
        # ----------------------------------------------------

        valid = (
            ~padding_mask
        ).unsqueeze(-1)


        x = x * valid


        summed = x.sum(
            dim=1
        )


        number_of_particles = (
            valid.sum(
                dim=1
            )
        )


        event_representation = (
            summed
            / number_of_particles
        )


        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        output = self.classifier(
            event_representation
        )


        return output


# ============================================================
# 9. Create model
# ============================================================

model = ParticleTransformer(
    input_dim=3,
    d_model=64,
    n_heads=4,
    n_layers=3,
    dropout=0.1
).to(device)


print(model)


# ============================================================
# 10. Loss and optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)


# ============================================================
# 11. Training
# ============================================================

epochs = 15


for epoch in range(epochs):

    model.train()

    total_loss = 0.0

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


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        predictions = model(
            particles,
            padding_mask
        )


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            predictions,
            labels
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        total_loss += loss.item()


        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        predicted_class = (
            predictions.argmax(
                dim=1
            )
        )


        correct += (
            predicted_class == labels
        ).sum().item()


        total += labels.size(0)


    accuracy = (
        correct / total
    )


    average_loss = (
        total_loss
        / len(train_loader)
    )


    print(
        f"Epoch {epoch + 1:02d}/{epochs} "
        f"Loss = {average_loss:.4f} "
        f"Accuracy = {accuracy:.4f}"
    )


# ============================================================
# 12. Testing
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


        predictions = model(
            particles,
            padding_mask
        )


        predicted_class = (
            predictions.argmax(
                dim=1
            )
        )


        correct += (
            predicted_class == labels
        ).sum().item()


        total += labels.size(0)


test_accuracy = (
    correct / total
)


print(
    "\nTest accuracy:",
    test_accuracy
)


# ============================================================
# 13. Test permutation invariance
# ============================================================

model.eval()


event, label = test_dataset[0]


# Original event

original_event = event.unsqueeze(0)


original_mask = torch.zeros(
    1,
    event.shape[0],
    dtype=torch.bool
)


# Shuffle particles

permutation = torch.randperm(
    event.shape[0]
)


shuffled_event = (
    event[permutation]
    .unsqueeze(0)
)


shuffled_mask = torch.zeros(
    1,
    event.shape[0],
    dtype=torch.bool
)


original_event = original_event.to(
    device
)

original_mask = original_mask.to(
    device
)

shuffled_event = shuffled_event.to(
    device
)

shuffled_mask = shuffled_mask.to(
    device
)


with torch.no_grad():

    original_output = model(
        original_event,
        original_mask
    )

    shuffled_output = model(
        shuffled_event,
        shuffled_mask
    )


print(
    "\nOriginal prediction:",
    torch.softmax(
        original_output,
        dim=1
    )
)


print(
    "Shuffled prediction:",
    torch.softmax(
        shuffled_output,
        dim=1
    )
)
