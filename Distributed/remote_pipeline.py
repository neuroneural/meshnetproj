from coinstac_computation import COINSTACPyNode, ComputationPhase, PhaseEndWithSuccess
import numpy as np
import torch

class PhaseAggregateMatrix(ComputationPhase):
    def compute(self):
        out = {}
        data = self.recv("site_matrix")
        aggregated_gradients = []
        for grad_list in zip(*data):
            aggregated_grad = torch.stack(grad_list).mean(dim=0)
            aggregated_gradients.append(aggregated_grad)

        #mean_data = np.array(data).mean(0)
        out.update(**self.send("averaged_matrix", aggregated_gradients))
        return out


remote = COINSTACPyNode(mode='remote', debug=True)
remote.add_phase(PhaseAggregateMatrix)
remote.add_phase(PhaseEndWithSuccess)
