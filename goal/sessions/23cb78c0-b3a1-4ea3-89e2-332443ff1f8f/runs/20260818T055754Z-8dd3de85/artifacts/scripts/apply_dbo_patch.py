import os
SP='/shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm'
def edit(rel, pairs):
    p=os.path.join(SP,rel)
    if not os.path.exists(p+'.orig0005'):
        open(p+'.orig0005','w').write(open(p).read())
    s=open(p).read()
    for old,new in pairs:
        assert old in s, (rel, old[:60]); s=s.replace(old,new,1)
    open(p,'w').write(s)
edit('config/vllm.py', [('''        if self.parallel_config.use_ubatching:
            a2a_backend = self.parallel_config.all2all_backend
            assert a2a_backend in [
                "deepep_low_latency",
                "deepep_high_throughput",
                "nixl_ep",
            ], (''','''        if self.parallel_config.use_ubatching and (
            self.model_config is None or self.model_config.is_moe
        ):
            # Dense models overlap TP all-reduces instead of all2all
            # dispatch/combine, so they need no special all2all backend.
            a2a_backend = self.parallel_config.all2all_backend
            assert a2a_backend in [
                "deepep_low_latency",
                "deepep_high_threshold",
                "nixl_ep",
            ], (''')])
# fix typo introduced above deliberately? no - keep exact names
s=open(os.path.join(SP,'config/vllm.py')).read().replace('"deepep_high_threshold"','"deepep_high_throughput"'); open(os.path.join(SP,'config/vllm.py'),'w').write(s)
edit('distributed/parallel_state.py', [('''    def _all_reduce_out_place(self, input_: torch.Tensor) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.all_reduce(input_)
''','''    def _all_reduce_out_place(self, input_: torch.Tensor) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        from vllm.v1.worker.ubatching import (
            dbo_enabled,
            dbo_yield_and_switch_from_comm_to_compute,
            dbo_yield_and_switch_from_compute_to_comm,
        )

        if dbo_enabled():
            # Dual-batch overlap for dense TP: hand the compute stream to the
            # other micro-batch while this one's all-reduce runs on the comm
            # stream, then wait for it before continuing.
            dbo_yield_and_switch_from_compute_to_comm()
            output = self.device_communicator.all_reduce(input_)
            dbo_yield_and_switch_from_comm_to_compute()
            return output
        return self.device_communicator.all_reduce(input_)
''')])
print('applied')
