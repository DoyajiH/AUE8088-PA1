from torchmetrics import Metric
import torch
import src.config as cfg

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes: int = cfg.NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        # TP, FP, FN 
        self.add_state("tp", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.argmax(preds, dim=1)
        # check if preds and target have equal shape
        if preds.shape != target.shape:
            raise ValueError(f"Shape mismatch: preds {preds.shape} vs target {target.shape}")
        # accumulate one-vs-rest TP, FP, FN 
        for c in range(self.num_classes):
            pred_c = (preds == c)
            true_c = (target == c)
            self.tp[c] += torch.logical_and(pred_c, true_c).sum()
            self.fp[c] += torch.logical_and(pred_c, ~true_c).sum()
            self.fn[c] += torch.logical_and(~pred_c, true_c).sum()

    def compute(self):
        # calculate precision, recall
        precision = self.tp.float() / (self.tp + self.fp).float().clamp(min=1e-6)
        recall    = self.tp.float() / (self.tp + self.fn).float().clamp(min=1e-6)
        # calculate per-class F1, macro-F1
        f1_per_class = 2 * (precision * recall) / (precision + recall).clamp(min=1e-6)
        macro_f1      = f1_per_class.mean()
        return f1_per_class, macro_f1

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        if preds.shape != target.shape:
            raise ValueError(f"Shape mismatch: preds {preds.shape} vs target {target.shape}")

        # [TODO] Count the number of correct prediction
        correct = (preds == target).sum()

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
