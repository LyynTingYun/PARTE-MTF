import torch
import torch.nn as nn


class SeasonDecompose_Mask:

    def __init__(
        self,
        length,
        kernel_size,
        segment_len,
        p_tmask=0.2,
        topk=3
    ):
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.l = length
        self.segment_len = segment_len

        self.equal_last_seg = (length % segment_len == 0)

        self.n_tseg = (
            length // segment_len
            if self.equal_last_seg
            else length // segment_len + 1
        )

        self.p_mask = p_tmask

        self.trend_extractor = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            count_include_pad=False
        )

    def extract_season_trend(self, x):
        nan_mask = ~x.isnan().any(dim=-1)
        x = x.clone()
        x[~nan_mask] = 0

        xt = self.trend_extractor(
            x.transpose(-1, -2)
        ).transpose(-1, -2)

        if self.kernel_size % 2 == 0:
            xt = xt[:, :-1]
        xs = x - xt

        return xt, xs

    def _mask_trend(self, xt):
        """
        xt: (B, L, d)
        """
        mask = torch.rand_like(xt) < self.p_mask

        xt_masked = xt.clone()
        xt_masked[mask] = 0

        return xt_masked

    def _mask_season(self, xs):
        """
        xs: (B, L, d)

        Returns
        -------
        masked_xs_list : list[(B, L, d)]
        """
        B, L, d = xs.shape

        assert L % self.segment_len == 0, (
            f"L ({L}) must be divisible by "
            f"segment_len ({self.segment_len})"
        )

        season_len = self.segment_len

        masked_xs_list = []

        for season_idx in range(self.n_tseg):
            xs_masked = xs.clone()
            start = season_idx * season_len
            end = min((season_idx + 1) * season_len, L)
            xs_masked[:, start:end, :] = 0
            masked_xs_list.append(xs_masked)

        return masked_xs_list

    def mask(self, xt, xs):

        masked_xt = self._mask_trend(xt)

        masked_xs_list = self._mask_season(xs)

        return masked_xt, masked_xs_list