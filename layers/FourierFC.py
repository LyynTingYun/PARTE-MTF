import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeasonalFrequencyMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        seq_len,

        # Ablation switches
        use_fft=True,
        use_freq_weight=True,
        use_residual=True,
        use_complex_activation=True,
        use_fc=True,
        activation="relu"
    ):
        super(SeasonalFrequencyMLP, self).__init__()

        self.length = seq_len
        self.n_freq = seq_len // 2 + 1
        self.hidden_dim = hidden_dim

        self.use_fft = use_fft
        self.use_freq_weight = use_freq_weight
        self.use_residual = use_residual
        self.use_complex_activation = use_complex_activation
        self.use_fc = use_fc
        self.activation = activation

        # =====================================================
        # Frequency-specific weights
        # =====================================================
        self.W = nn.Parameter(
            torch.empty(
                (self.n_freq, input_dim, hidden_dim),
                dtype=torch.cfloat
            )
        )

        self.B = nn.Parameter(
            torch.empty(
                (self.n_freq, hidden_dim),
                dtype=torch.cfloat
            )
        )

        # =====================================================
        # Shared frequency weight (ablation)
        # =====================================================
        self.W_shared = nn.Parameter(
            torch.empty(
                (input_dim, hidden_dim),
                dtype=torch.cfloat
            )
        )

        # =====================================================
        # Residual branch
        # =====================================================
        self.W_rc = nn.Parameter(
            torch.empty((input_dim, hidden_dim))
        )

        # =====================================================
        # Prediction head
        # =====================================================
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.output_projection = nn.Linear(
            hidden_dim,
            output_dim
        )

        self._reset_parameters()

    def _reset_parameters(self):

        nn.init.kaiming_uniform_(self.W.real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W.imag, a=math.sqrt(5))

        nn.init.kaiming_uniform_(self.W_shared.real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_shared.imag, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
            self.W.real
        )

        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0

        nn.init.uniform_(self.B.real, -bound, bound)
        nn.init.uniform_(self.B.imag, -bound, bound)

        nn.init.kaiming_uniform_(self.W_rc, a=math.sqrt(5))

    def _complex_activation(self, x):

        if not self.use_complex_activation:
            return x

        if self.activation == "relu":

            return torch.complex(
                F.relu(x.real),
                F.relu(x.imag)
            )

        elif self.activation == "gelu":

            return torch.complex(
                F.gelu(x.real),
                F.gelu(x.imag)
            )

        else:
            raise ValueError(
                f"Unsupported activation: {self.activation}"
            )

    def _frequency_forward(self, x):

        if self.use_freq_weight:

            out = (
                torch.einsum(
                    'bfi,fio->bfo',
                    x,
                    self.W
                )
                + self.B
            )

        else:
            # Shared spectral weight ablation
            out = (
                torch.einsum(
                    'bfi,io->bfo',
                    x,
                    self.W_shared
                )
                + self.B
            )

        out = self._complex_activation(out)

        return out

    def forward(self, x):

        # ---------------------------------------------
        # Residual branch
        # ---------------------------------------------
        if self.use_residual:

            x_rc = torch.einsum(
                'btd,dk->btk',
                x,
                self.W_rc
            )

        # ---------------------------------------------
        # Frequency branch
        # ---------------------------------------------
        if self.use_fft:

            x_fft = torch.fft.rfft(
                x,
                dim=1
            )[:, :self.n_freq]

            out_fft = self._frequency_forward(
                x_fft
            )

            out = torch.fft.irfft(
                out_fft,
                n=x.size(1),
                dim=1
            )

        else:
            # FFT branch removed
            out = torch.zeros(
                x.size(0),
                x.size(1),
                self.hidden_dim,
                device=x.device,
                dtype=x.dtype
            )

        # ---------------------------------------------
        # Residual merge
        # ---------------------------------------------
        if self.use_residual:
            out = out + x_rc

        # ---------------------------------------------
        # Prediction head
        # ---------------------------------------------
        if self.use_fc:
            out = self.fc(out)
        else:
            out = self.output_projection(out)

        return out


if __name__ == "__main__":

    torch.manual_seed(42)

    # Simulated data
    x = torch.randn(1, 336, 512)

    configs = [
        # Full model
        {
            "name": "Full",
            "use_fft": True,
            "use_freq_weight": True,
            "use_residual": True,
            "use_complex_activation": True,
            "use_fc": True,
        },

        # Remove FFT
        {
            "name": "WoFFT",
            "use_fft": False,
            "use_freq_weight": True,
            "use_residual": True,
            "use_complex_activation": True,
            "use_fc": True,
        },

        # Shared spectral weight
        {
            "name": "WoFreqWeight",
            "use_fft": True,
            "use_freq_weight": False,
            "use_residual": True,
            "use_complex_activation": True,
            "use_fc": True,
        },

        # Remove residual
        {
            "name": "WoResidual",
            "use_fft": True,
            "use_freq_weight": True,
            "use_residual": False,
            "use_complex_activation": True,
            "use_fc": True,
        },

        # Remove complex activation
        {
            "name": "WoComplexAct",
            "use_fft": True,
            "use_freq_weight": True,
            "use_residual": True,
            "use_complex_activation": False,
            "use_fc": True,
        },

        # Linear head only
        {
            "name": "WoFC",
            "use_fft": True,
            "use_freq_weight": True,
            "use_residual": True,
            "use_complex_activation": True,
            "use_fc": False,
        },

        # FFT only
        {
            "name": "FFTOnly",
            "use_fft": True,
            "use_freq_weight": True,
            "use_residual": False,
            "use_complex_activation": True,
            "use_fc": False,
        },

        # Residual only
        {
            "name": "ResidualOnly",
            "use_fft": False,
            "use_freq_weight": True,
            "use_residual": True,
            "use_complex_activation": True,
            "use_fc": False,
        },

        # Minimal model
        {
            "name": "Minimal",
            "use_fft": False,
            "use_freq_weight": False,
            "use_residual": False,
            "use_complex_activation": False,
            "use_fc": False,
        },
    ]

    for cfg in configs:

        print("=" * 80)
        print("Testing:", cfg["name"])

        try:

            model = SeasonalFrequencyMLP(
                input_dim=512,
                hidden_dim=256,
                output_dim=128,
                seq_len=336,

                use_fft=cfg["use_fft"],
                use_freq_weight=cfg["use_freq_weight"],
                use_residual=cfg["use_residual"],
                use_complex_activation=cfg["use_complex_activation"],
                use_fc=cfg["use_fc"],
            )

            y = model(x)

            print("Input shape :", x.shape)
            print("Output shape:", y.shape)
            print("PASS")

        except Exception as e:

            print("FAIL")
            print(type(e).__name__, e)

    print("\nAll tests finished.")