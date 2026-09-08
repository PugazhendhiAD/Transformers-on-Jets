import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# 1. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)


# ============================================================
# 2. Create an irregularly sampled time series
# ============================================================

torch.manual_seed(42)

# Generate random time intervals
dt = 0.05 + 0.20 * torch.rand(10000)

# Actual observation times
t = torch.cumsum(dt, dim=0)

# Sine wave
y = torch.sin(t)


# ============================================================
# 3. Create sequences
# ============================================================

sequence_length = 60

X_time = []
X_value = []
Y = []

for i in range(len(y) - sequence_length):

    # Actual times of the 50 observations
    X_time.append(
        t[i:i + sequence_length]
    )

    # Observed values
    X_value.append(
        y[i:i + sequence_length]
    )

    # Value we want to predict
    Y.append(
        y[i + sequence_length]
    )


X_time = torch.stack(X_time)
X_value = torch.stack(X_value)
Y = torch.stack(Y)


print("Time shape:", X_time.shape)
print("Value shape:", X_value.shape)
print("Target shape:", Y.shape)


# ============================================================
# 4. Train / test split
# ============================================================

split = int(0.8 * len(X_time))

X_time_train = X_time[:split]
X_value_train = X_value[:split]
Y_train = Y[:split]

X_time_test = X_time[split:]
X_value_test = X_value[split:]
Y_test = Y[split:]


# Move to GPU if available

X_time_train = X_time_train.to(device)
X_value_train = X_value_train.to(device)
Y_train = Y_train.to(device)

X_time_test = X_time_test.to(device)
X_value_test = X_value_test.to(device)
Y_test = Y_test.to(device)


# ============================================================
# 5. Continuous Fourier Time Encoding
# ============================================================

class TimeEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_period=1000.0
    ):

        super().__init__()

        # Number of frequencies
        n_frequencies = d_model // 2

        # Frequencies distributed logarithmically
        frequencies = torch.exp(
            torch.linspace(
                0,
                math.log(max_period),
                n_frequencies
            )
        )

        self.register_buffer(
            "frequencies",
            frequencies
        )


    def forward(self, t):

        # ----------------------------------------------------
        # t shape:
        #
        # (batch, sequence_length)
        #
        # frequencies shape:
        #
        # (d_model / 2)
        # ----------------------------------------------------

        angles = (
            t.unsqueeze(-1)
            / self.frequencies
        )


        # Sin and cosine features

        sin_features = torch.sin(
            angles
        )

        cos_features = torch.cos(
            angles
        )


        # Combine them

        encoding = torch.cat(
            [
                sin_features,
                cos_features
            ],
            dim=-1
        )


        return encoding


# ============================================================
# 6. Transformer model
# ============================================================

class IrregularTimeTransformer(nn.Module):

    def __init__(
        self,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    ):

        super().__init__()


        # ----------------------------------------------------
        # Encode the observed value
        # ----------------------------------------------------

        self.value_embedding = nn.Linear(
            1,
            d_model
        )


        # ----------------------------------------------------
        # Continuous time encoding
        # ----------------------------------------------------

        self.time_encoding = TimeEncoding(
            d_model
        )


        # ----------------------------------------------------
        # Transformer encoder
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )


        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )


        # ----------------------------------------------------
        # Prediction head
        # ----------------------------------------------------

        self.output_layer = nn.Linear(
            d_model,
            1
        )


    def forward(
        self,
        values,
        times
    ):

        # ----------------------------------------------------
        # values:
        #
        # (batch, sequence_length)
        #
        # times:
        #
        # (batch, sequence_length)
        # ----------------------------------------------------


        # Add feature dimension

        values = values.unsqueeze(-1)


        # ----------------------------------------------------
        # Value embedding
        # ----------------------------------------------------

        value_embedding = self.value_embedding(
            values
        )


        # ----------------------------------------------------
        # Time encoding
        # ----------------------------------------------------

        time_embedding = self.time_encoding(
            times
        )


        # ----------------------------------------------------
        # Combine value and time
        # ----------------------------------------------------

        x = (
            value_embedding
            + time_embedding
        )


        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        x = self.transformer(
            x
        )


        # ----------------------------------------------------
        # Take representation of final observation
        # ----------------------------------------------------

        x = x[:, -1, :]


        # ----------------------------------------------------
        # Predict next value
        # ----------------------------------------------------

        x = self.output_layer(
            x
        )


        return x.squeeze(-1)


# ============================================================
# 7. Create model
# ============================================================

model = IrregularTimeTransformer(
    d_model=64,
    n_heads=4,
    n_layers=2,
    dropout=0.1
).to(device)


print(model)


# ============================================================
# 8. Loss and optimizer
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)


# ============================================================
# 9. Training
# ============================================================

epochs = 20

batch_size = 128


for epoch in range(epochs):

    model.train()

    total_loss = 0.0


    # Shuffle the training examples

    permutation = torch.randperm(
        X_time_train.size(0),
        device=device
    )


    for i in range(
        0,
        X_time_train.size(0),
        batch_size
    ):

        indices = permutation[
            i:i + batch_size
        ]


        time_batch = X_time_train[
            indices
        ]

        value_batch = X_value_train[
            indices
        ]

        target_batch = Y_train[
            indices
        ]


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        prediction = model(
            value_batch,
            time_batch
        )


        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = criterion(
            prediction,
            target_batch
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        total_loss += loss.item()


    average_loss = (
        total_loss
        / (X_time_train.size(0) // batch_size)
    )


    print(
        f"Epoch {epoch + 1:02d}/{epochs} "
        f"Loss = {average_loss:.6f}"
    )


# ============================================================
# 10. Test
# ============================================================

model.eval()

with torch.no_grad():

    predictions = model(
        X_value_test,
        X_time_test
    )


test_loss = criterion(
    predictions,
    Y_test
)


print(
    "\nTest MSE:",
    test_loss.item()
)


# ============================================================
# 11. Plot predictions
# ============================================================

predictions_cpu = predictions.cpu()

Y_test_cpu = Y_test.cpu()


plt.figure(
    figsize=(12, 5)
)


plt.plot(
    Y_test_cpu[:500].numpy(),
    label="True"
)


plt.plot(
    predictions_cpu[:500].numpy(),
    label="Predicted"
)


plt.xlabel("Sample")

plt.ylabel("Value")

plt.title(
    "Transformer with Continuous Time Encoding"
)

plt.legend()

plt.show()
