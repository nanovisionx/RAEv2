import torch as th


class Sampler:
    def __init__(self, transport, guidance_config):
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.guidance_config = guidance_config
        self.omega = guidance_config.cfg.scale
        self.t_start = guidance_config.cfg.t_min
        self.t_end = guidance_config.cfg.t_max

    def sample_ode(self, *, num_steps=50):
        t_grid = th.linspace(1.0, 0.0, num_steps + 1)
        shift = self.transport.time_dist_shift
        t_grid = shift * t_grid / (1 + (shift - 1) * t_grid)

        def sample_fn(x, model, **model_kwargs):
            device = x.device
            t_steps = t_grid.to(device)
            B = x.shape[0]

            model_kwargs_ = model_kwargs.copy()
            for k, v in (('omega', self.omega), ('t_start', self.t_start), ('t_end', self.t_end)):
                if v is not None:
                    model_kwargs_[k] = th.full((B,), v, device=device)

            for i in range(num_steps):
                h = t_steps[i] - t_steps[i + 1]
                t_batch = th.full((B,), t_steps[i].item(), device=device)
                d_cur = self.drift(x, t_batch, model, **model_kwargs_)
                x = x - h * d_cur

            return x.unsqueeze(0)

        return sample_fn
