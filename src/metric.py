from torchmetrics import Metric
import torch
import src.config as cfg

# [TODO] Implement this!
class MyF1Score(Metric):
    """
    one-vs-rest per-class F1 + macro-F1
    update()로 배치별 TP/FP/FN 누적,
    compute()로 per-class F1 벡터와 macro-F1 스칼라 반환
    """
    def __init__(self, num_classes: int = cfg.NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        # 클래스별 TP, FP, FN 상태 등록
        self.add_state("tp", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # 1) 로짓 → 예측 클래스 인덱스
        preds = torch.argmax(preds, dim=1)
        # 2) shape mismatch 방지
        if preds.shape != target.shape:
            raise ValueError(f"Shape mismatch: preds {preds.shape} vs target {target.shape}")
        # 3) one-vs-rest 방식으로 클래스별 TP/FP/FN 누적
        for c in range(self.num_classes):
            pred_c = (preds == c)
            true_c = (target == c)
            self.tp[c] += torch.logical_and(pred_c, true_c).sum()
            self.fp[c] += torch.logical_and(pred_c, ~true_c).sum()
            self.fn[c] += torch.logical_and(~pred_c, true_c).sum()

    def compute(self):
        # 4) Precision, Recall 계산
        precision = self.tp.float() / (self.tp + self.fp).float().clamp(min=1e-6)
        recall    = self.tp.float() / (self.tp + self.fn).float().clamp(min=1e-6)
        # 5) per-class F1, macro-F1 계산
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
