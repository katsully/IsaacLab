# road_utils.py

import torch

class RoadCorridor:
    """
    Defines a road as a polyline centerline with a half-width.
    Provides: spawn sampling, distance queries, heading queries.
    """

    def __init__(
        self,
        waypoints: list[tuple[float, float, float]],
        half_width: float = 2.0,
        device: str = "cuda",
    ):
        self.device = device
        self.half_width = half_width

        wp = torch.tensor(waypoints, dtype=torch.float32, device=device)
        self.points_xy = wp[:, :2]
        self.heights = wp[:, 2]
        self.num_segments = wp.shape[0] - 1

        # Pre-compute segment geometry
        self.seg_starts = self.points_xy[:-1]
        self.seg_ends = self.points_xy[1:]
        self.seg_dirs = self.seg_ends - self.seg_starts
        self.seg_lengths = torch.norm(self.seg_dirs, dim=1)
        self.seg_unit = self.seg_dirs / self.seg_lengths.unsqueeze(1).clamp(min=1e-6)

        # Cumulative arc-length for uniform sampling
        self.cum_len = torch.cat([
            torch.zeros(1, device=device),
            torch.cumsum(self.seg_lengths, dim=0),
        ])
        self.total_length = self.cum_len[-1].item()

    def distance_to_centerline(self, pos_xy: torch.Tensor) -> torch.Tensor:
        """Shortest distance from each (N,2) point to the polyline. Returns (N,)."""
        N = pos_xy.shape[0]
        S = self.num_segments

        # Reshape for broadcasting
        p = pos_xy.unsqueeze(1).expand(N, S, 2)
        a = self.seg_starts.unsqueeze(0).expand(N, S, 2)
        ab = self.seg_dirs.unsqueeze(0).expand(N, S, 2)
        L2 = (self.seg_lengths ** 2).unsqueeze(0).expand(N, S)

        # Project point p onto segment ab
        ap = p - a
        t = (ap * ab).sum(dim=2) / L2.clamp(min=1e-8)
        t = t.clamp(0.0, 1.0)

        closest = a + t.unsqueeze(2) * ab
        dist = torch.norm(p - closest, dim=2)
        min_dist, _ = dist.min(dim=1)
        return min_dist

    def is_on_road(self, pos_xy: torch.Tensor) -> torch.Tensor:
        """(N,) bool — True if within half_width of centerline."""
        return self.distance_to_centerline(pos_xy) < self.half_width

    def heading_at(self, pos_xy: torch.Tensor) -> torch.Tensor:
        """Yaw angle (rad) of the road at the closest segment. Returns (N,)."""
        N = pos_xy.shape[0]
        S = self.num_segments

        p = pos_xy.unsqueeze(1).expand(N, S, 2)
        a = self.seg_starts.unsqueeze(0).expand(N, S, 2)
        ab = self.seg_dirs.unsqueeze(0).expand(N, S, 2)
        L2 = (self.seg_lengths ** 2).unsqueeze(0).expand(N, S)

        ap = p - a
        t = (ap * ab).sum(dim=2) / L2.clamp(min=1e-8)
        t = t.clamp(0.0, 1.0)
        closest = a + t.unsqueeze(2) * ab
        dist = torch.norm(p - closest, dim=2)
        _, best_seg = dist.min(dim=1)

        d = self.seg_unit[best_seg]
        return torch.atan2(d[:, 1], d[:, 0])

    def height_at(self, pos_xy: torch.Tensor) -> torch.Tensor:
        """Interpolated road-surface Z at closest segment. Returns (N,)."""
        N = pos_xy.shape[0]
        S = self.num_segments

        p = pos_xy.unsqueeze(1).expand(N, S, 2)
        a = self.seg_starts.unsqueeze(0).expand(N, S, 2)
        ab = self.seg_dirs.unsqueeze(0).expand(N, S, 2)
        L2 = (self.seg_lengths ** 2).unsqueeze(0).expand(N, S)

        ap = p - a
        t = (ap * ab).sum(dim=2) / L2.clamp(min=1e-8)
        t = t.clamp(0.0, 1.0)
        closest = a + t.unsqueeze(2) * ab
        dist = torch.norm(p - closest, dim=2)
        _, best_seg = dist.min(dim=1)

        t_best = t.gather(1, best_seg.unsqueeze(1)).squeeze(1)
        h0 = self.heights[best_seg]
        h1 = self.heights[best_seg + 1]
        return h0 + t_best * (h1 - h0)

    def sample_positions(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample n random (x, y, z) on the road + yaw heading.
        Returns: pos (n,3), yaw (n,)
        """
        s = torch.rand(n, device=self.device) * self.total_length

        seg_idx = torch.searchsorted(self.cum_len[1:], s).clamp(0, self.num_segments - 1)
        local_s = s - self.cum_len[seg_idx]
        t = (local_s / self.seg_lengths[seg_idx].clamp(min=1e-6)).clamp(0.0, 1.0)

        # Centerline XY
        center = self.seg_starts[seg_idx] + t.unsqueeze(1) * self.seg_dirs[seg_idx]

        # Lateral offset (perpendicular) — stay within 70% of road width
        lat = (torch.rand(n, device=self.device) * 2 - 1) * self.half_width * 0.7
        perp = torch.stack([
            -self.seg_unit[seg_idx, 1],
             self.seg_unit[seg_idx, 0],
        ], dim=1)
        xy = center + lat.unsqueeze(1) * perp

        # Height
        h0 = self.heights[seg_idx]
        h1 = self.heights[seg_idx + 1]
        z = h0 + t * (h1 - h0)

        pos = torch.stack([xy[:, 0], xy[:, 1], z], dim=1)

        # Heading
        yaw = torch.atan2(
            self.seg_unit[seg_idx, 1],
            self.seg_unit[seg_idx, 0],
        )

        return pos, yaw