import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================
# 1. Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)


# ============================================================
# 2. Create a sine-wave dataset
# ============================================================

torch.manual_seed(42)

# Generate the time axis
t = torch.linspace(0, 100, 10000)

# Sine wave
y = torch.sin(t)


# ============================================================
# 3. Convert the sine wave into sequences
# ============================================================

sequence_length = 50

X = []
Y = []

for i in range(len(y) - sequence_length):

    # Input sequence
    X.append(y[i:i + sequence_length])

    # Value immediately after the sequence
    Y.append(y[i + sequence_length])


X = torch.stack(X)
Y = torch.stack(Y)

print("X shape:", X.shape)
print("Y shape:", Y.shape)


# ============================================================
# 4. Train / test split
# ============================================================

split = int(0.8 * len(X))

X_train = X[:split]
Y_train = Y[:split]

X_test = X[split:]
Y_test = Y[split:]


# Move to GPU if available
X_train = X_train.to(device)
Y_train = Y_train.to(device)

X_test = X_test.to(device)
Y_test = Y_test.to(device)


# ============================================================
# 5. Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_length=5000):

        super().__init__()

        position = torch.arange(max_length).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_length, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape:
        # (1, max_length, d_model)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)


    def forward(self, x):

        # x shape:
        # (batch, sequence_length, d_model)

        return x + self.pe[:, :x.size(1)]


# ============================================================
# 6. Transformer model
# ============================================================

class SineTransformer(nn.Module):

    def __init__(
        self,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    ):

        super().__init__()


        # ----------------------------------------------------
        # Convert scalar sine value into d_model dimensions
        # ----------------------------------------------------

        self.input_embedding = nn.Linear(1, d_model)


        # ----------------------------------------------------
        # Positional encoding
        # ----------------------------------------------------

        self.positional_encoding = PositionalEncoding(
            d_model
        )


        # ----------------------------------------------------
        # Transformer Encoder
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
        # Output layer
        # ----------------------------------------------------

        self.output_layer = nn.Linear(d_model, 1)


    def forward(self, x):

        # x:
        # (batch, sequence_length)

        # Add feature dimension
        x = x.unsqueeze(-1)

        # Now:
        # (batch, sequence_length, 1)


        # Convert scalar → vector
        x = self.input_embedding(x)

        # (batch, sequence_length, d_model)


        # Add positional information
        x = self.positional_encoding(x)


        # Transformer
        x = self.transformer(x)

        # (batch, sequence_length, d_model)


        # Take the representation of the final timestep
        x = x[:, -1, :]


        # Predict next value
        x = self.output_layer(x)

        return x.squeeze(-1)


# ============================================================
# 7. Create model
# ============================================================

model = SineTransformer(
    d_model=64,
    n_heads=4,
    n_layers=2
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


    # Shuffle training examples
    permutation = torch.randperm(
        X_train.size(0),
        device=device
    )


    for i in range(0, X_train.size(0), batch_size):

        indices = permutation[i:i + batch_size]

        x_batch = X_train[indices]

        y_batch = Y_train[indices]


        # Forward pass
        prediction = model(x_batch)


        # Loss
        loss = criterion(
            prediction,
            y_batch
        )


        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        total_loss += loss.item()


    average_loss = total_loss / (
        X_train.size(0) // batch_size
    )


    print(
        f"Epoch {epoch + 1:02d}/{epochs} "
        f"Loss = {average_loss:.6f}"
    )


# ============================================================
# 10. Test the model
# ============================================================

model.eval()

with torch.no_grad():

    predictions = model(X_test)


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


plt.figure(figsize=(12, 5))

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

plt.title("Transformer: Sine Wave Prediction")

plt.legend()

plt.show()