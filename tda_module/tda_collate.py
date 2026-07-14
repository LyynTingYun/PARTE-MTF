import torch
import torch.nn.functional as F

class TDACollateFn:

    def __call__(self, batch):

        result = {}

        max_points = max(
            sample["pd_tensor"].shape[-2]
            for sample in batch
        )

        pd_tensor_list = []
        pd_mask_list = []

        for sample in batch:

            pd_tensor = sample["pd_tensor"]
            pd_mask = sample["pd_mask"]

            cur_points = pd_tensor.shape[-2]

            pad_points = max_points - cur_points

            if pad_points > 0:

                # pd_tensor:
                # (..., max_points, 2)

                pd_tensor = F.pad(
                    pd_tensor,
                    (0, 0,          # last dim (2)
                     0, pad_points) # point dimension
                )

                # pd_mask:
                # (..., max_points)

                pd_mask = F.pad(
                    pd_mask,
                    (0, pad_points)
                )

            pd_tensor_list.append(pd_tensor)
            pd_mask_list.append(pd_mask)

        result["pd_tensor"] = torch.stack(pd_tensor_list)
        result["pd_mask"] = torch.stack(pd_mask_list)

        keys = batch[0].keys()

        for key in keys:

            if key in ["pd_tensor", "pd_mask"]:
                continue

            values = [sample[key] for sample in batch]

            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = values

        result["max_points"] = max_points

        return result