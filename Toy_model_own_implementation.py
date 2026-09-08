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
# 2. Create sine-wave dataset
# ============================================================

torch.manual_seed(42)

t = torch.linspace(0, 100, 10000)

y = torch.sin(t)


# ============================================================
# 3. Create sequences
# ============================================================

sequence_length = 50

X = []
Y = []

for i in range(len(y) - sequence_length):

    X.append(
        y[i:i + sequence_length]
    )

    Y.append(
        y[i + sequence_length]
    )


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

X_train = X_train.to(device)
Y_train = Y_train.to(device)

X_test = X_test.to(device)
Y_test = Y_test.to(device)


# ============================================================
# 5. Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_length=5000
    ):

        super().__init__()

        position = torch.arange(
            max_length
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            )
            * (
                -math.log(10000.0)
                / d_model
            )
        )

        pe = torch.zeros(
            max_length,
            d_model
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )


    def forward(self, x):

        return (
            x
            + self.pe[:, :x.size(1)]
        )


# ============================================================
# 6. Multi-Head Self Attention
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads
    ):

        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model

        self.n_heads = n_heads

        self.head_dim = (
            d_model // n_heads
        )


        # Q, K, V projections

        self.W_q = nn.Linear(
            d_model,
            d_model
        )

        self.W_k = nn.Linear(
            d_model,
            d_model
        )

        self.W_v = nn.Linear(
            d_model,
            d_model
        )


        # Output projection

        self.W_o = nn.Linear(
            d_model,
            d_model
        )


    def forward(self, x):

        # ----------------------------------------------------
        # x shape:
        #
        # (batch, sequence_length, d_model)
        # ----------------------------------------------------

        batch_size = x.size(0)

        sequence_length = x.size(1)


        # ----------------------------------------------------
        # Create Q, K, V
        # ----------------------------------------------------

        Q = self.W_q(x)

        K = self.W_k(x)

        V = self.W_v(x)


        # ----------------------------------------------------
        # Split into attention heads
        #
        # Before:
        #
        # batch × sequence × d_model
        #
        # After:
        #
        # batch × heads × sequence × head_dim
        # ----------------------------------------------------

        Q = Q.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim
        )

        K = K.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim
        )

        V = V.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim
        )


        Q = Q.transpose(1, 2)

        K = K.transpose(1, 2)

        V = V.transpose(1, 2)


        # ----------------------------------------------------
        # Attention scores
        #
        # Q K^T
        # ----------------------------------------------------

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )


        # ----------------------------------------------------
        # Scale by sqrt(d_k)
        # ----------------------------------------------------

        scores = scores / math.sqrt(
            self.head_dim
        )


        # ----------------------------------------------------
        # Convert scores into probabilities
        # ----------------------------------------------------

        attention = torch.softmax(
            scores,
            dim=-1
        )


        # ----------------------------------------------------
        # Weighted sum of V
        # ----------------------------------------------------

        output = torch.matmul(
            attention,
            V
        )


        # ----------------------------------------------------
        # Combine attention heads
        # ----------------------------------------------------

        output = output.transpose(
            1,
            2
        )

        output = output.contiguous()


        output = output.view(
            batch_size,
            sequence_length,
            self.d_model
        )


        # ----------------------------------------------------
        # Final linear projection
        # ----------------------------------------------------

        output = self.W_o(
            output
        )

        return output


# ============================================================
# 7. Feed Forward Network
# ============================================================

class FeedForward(nn.Module):

    def __init__(
        self,
        d_model,
        hidden_dim
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                d_model,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                d_model
            )
        )


    def forward(self, x):

        return self.network(x)


# ============================================================
# 8. Transformer Block
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1
    ):

        super().__init__()


        self.attention = MultiHeadAttention(
            d_model,
            n_heads
        )


        self.feed_forward = FeedForward(
            d_model,
            4 * d_model
        )


        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )


        self.dropout = nn.Dropout(
            dropout
        )


    def forward(self, x):

        # ====================================================
        # Attention
        # ====================================================

        attention_output = self.attention(x)


        # Residual connection

        x = x + self.dropout(
            attention_output
        )


        # Layer normalization

        x = self.norm1(x)


        # ====================================================
        # Feed Forward
        # ====================================================

        ff_output = self.feed_forward(x)


        # Residual connection

        x = x + self.dropout(
            ff_output
        )


        # Layer normalization

        x = self.norm2(x)


        return x


# ============================================================
# 9. Complete Transformer
# ============================================================

class ManualTransformer(nn.Module):

    def __init__(
        self,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    ):

        super().__init__()


        # ----------------------------------------------------
        # Input embedding
        # ----------------------------------------------------

        self.input_embedding = nn.Linear(
            1,
            d_model
        )


        # ----------------------------------------------------
        # Positional encoding
        # ----------------------------------------------------

        self.position = PositionalEncoding(
            d_model
        )


        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        self.blocks = nn.ModuleList([

            TransformerBlock(
                d_model,
                n_heads,
                dropout
            )

            for _ in range(n_layers)

        ])


        # ----------------------------------------------------
        # Prediction head
        # ----------------------------------------------------

        self.output_layer = nn.Linear(
            d_model,
            1
        )


    def forward(self, x):

        # ----------------------------------------------------
        # Input:
        #
        # (batch, sequence)
        # ----------------------------------------------------

        x = x.unsqueeze(-1)


        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------

        x = self.input_embedding(x)


        # ----------------------------------------------------
        # Positional encoding
        # ----------------------------------------------------

        x = self.position(x)


        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        for block in self.blocks:

            x = block(x)


        # ----------------------------------------------------
        # Take final token
        # ----------------------------------------------------

        x = x[:, -1, :]


        # ----------------------------------------------------
        # Predict next value
        # ----------------------------------------------------

        x = self.output_layer(x)


        return x.squeeze(-1)


# ============================================================
# 10. Create model
# ============================================================

model = ManualTransformer(
    d_model=64,
    n_heads=4,
    n_layers=2,
    dropout=0.1
).to(device)


print(model)


# ============================================================
# 11. Loss and optimizer
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)


# ============================================================
# 12. Training
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


    for i in range(
        0,
        X_train.size(0),
        batch_size
    ):

        indices = permutation[
            i:i + batch_size
        ]


        x_batch = X_train[
            indices
        ]

        y_batch = Y_train[
            indices
        ]


        # Forward

        prediction = model(
            x_batch
        )


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


    average_loss = (
        total_loss
        / (X_train.size(0) // batch_size)
    )


    print(
        f"Epoch {epoch + 1:02d}/{epochs} "
        f"Loss = {average_loss:.6f}"
    )


# ============================================================
# 13. Test
# ============================================================

model.eval()

with torch.no_grad():

    predictions = model(
        X_test
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
# 14. Plot
# ============================================================

predictions = predictions.cpu()

Y_test_plot = Y_test.cpu()


plt.figure(
    figsize=(12, 5)
)


plt.plot(
    Y_test_plot[:500].numpy(),
    label="True"
)


plt.plot(
    predictions[:500].numpy(),
    label="Predicted"
)


plt.xlabel("Sample")

plt.ylabel("Value")

plt.title(
    "Manual Transformer: "
    "Sine Wave Prediction"
)

plt.legend()

plt.show()