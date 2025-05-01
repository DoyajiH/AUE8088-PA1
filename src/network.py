# Python packages
import os
from termcolor import colored
from typing import Dict
import copy
from termcolor import colored

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models
from torchvision.models.alexnet import AlexNet
import torch

# Custom packages
from src.metric import MyAccuracy, MyF1Score
import src.config as cfg
from src.util import show_setting
from src.dataset import TinyImageNetDatasetModule


# [TODO: Optional] Rewrite this class if you want
class MyNetwork(AlexNet):
    def __init__(self):
        super().__init__()

        # [TODO] Modify feature extractor part in AlexNet


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [TODO: Optional] Modify this as well if you want
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str = 'resnet18',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
        ):
        super().__init__()

        # Network
        if model_name == 'MyNetwork':
            self.model = MyNetwork()
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metric
        self.accuracy = MyAccuracy()
        self.f1score = MyF1Score(num_classes)
        # validation epoch 마지막에 쓰기 위해 버퍼에 preds/targets 저장
        self._val_preds   = []
        self._val_targets = []

        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.pop('type')
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        self.log_dict({'loss/train': loss, 'accuracy/train': accuracy},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        preds    = torch.argmax(scores, dim=1)
        accuracy = self.accuracy(scores, y)
        f1_per_class, f1_macro = self.f1score(scores, y)
        self.log_dict({'loss/val': loss, 'accuracy/val': accuracy,'f1/val': f1_macro},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, scores, frequency = cfg.WANDB_IMG_LOG_FREQ)
        # 마지막 epoch에 출력할 confusion matrix용 버퍼에 저장
        self._val_preds.append(preds.cpu())
        self._val_targets.append(y.cpu())

    def on_validation_epoch_end(self):
        # 1) preds/targets를 모은 후 한 번에 꺼내기
        preds_all   = torch.cat(self._val_preds)
        targets_all = torch.cat(self._val_targets)

        # 2) confusion matrix 생성
        num_cls = self.f1score.num_classes
        cm = torch.zeros((num_cls, num_cls), dtype=torch.long)
        for p, t in zip(preds_all, targets_all):
            cm[p, t] += 1

        # 3) 클래스별 precision/recall/F1 계산
        tp       = cm.diag().float()
        fp       = cm.sum(dim=1).float() - tp
        fn       = cm.sum(dim=0).float() - tp
        precision= tp / (tp + fp).clamp(min=1e-6)
        recall   = tp / (tp + fn).clamp(min=1e-6)
        f1_cls   = 2 * (precision * recall) / (precision + recall).clamp(min=1e-6)

        # 4) 마지막 epoch일 때만 터미널 프린트
        if hasattr(self, "trainer") and self.current_epoch == self.trainer.max_epochs - 1:
            print("\n=== Confusion Matrix (first 10 classes) ===")
            print(cm[:10, :10])
            print("\n=== F1 Scores (first 10 classes) ===")
            for i in range(10):
                print(f" Class {i:3d}: F1 = {f1_cls[i]:.4f}")

        # 5) 다음 epoch을 위해 버퍼 초기화
        self._val_preds.clear()
        self._val_targets.clear()


    def _common_step(self, batch):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])
