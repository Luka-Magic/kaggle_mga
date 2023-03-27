from torch import nn

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
        # target_weight: (bs, n_joints)

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze() # (bs, hm_h*hm_w)
            print('hm_pred: ', heatmap_pred.shape, 'expect: (bs, hm_h*hm_w)')
            heatmap_gt = heatmaps_gt[idx].squeeze() # (bs, hm_h*hm_w)
            print('hm_gt: ', heatmap_gt.shape, 'expect: (bs, hm_h*hm_w)')
            print('hm_weight: ', target_weight.shape, 'expect: (bs, 196)')
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints