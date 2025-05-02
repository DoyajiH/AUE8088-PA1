import os
from lightning.pytorch import Trainer
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from src.dataset import TinyImageNetDatasetModule
from src.network import SimpleClassifier
import src.config as cfg

def main():
    # 실험 갯수는 세 리스트 중 가장 짧은 길이에 맞추기
    n = min(len(cfg.OPTIMIZER_LIST),
            len(cfg.SCHEDULER_LIST),
            len(cfg.MODEL_NAME_LIST))

    for i in range(n):
        # 1) 현재 실험 설정 덮어쓰기
        cfg.OPTIMIZER_PARAMS = cfg.OPTIMIZER_LIST[i]
        cfg.SCHEDULER_PARAMS = cfg.SCHEDULER_LIST[i]
        cfg.MODEL_NAME       = cfg.MODEL_NAME_LIST[i]

        # 2) WandB run 이름에 인덱스 붙이기
        cfg.WANDB_NAME          = f'{cfg.MODEL_NAME}-B{cfg.BATCH_SIZE}-{cfg.OPTIMIZER_PARAMS["type"]}'
        cfg.WANDB_NAME         += f'-{cfg.SCHEDULER_PARAMS["type"]}{cfg.OPTIMIZER_PARAMS["lr"]:.1E}'
        wandb_logger = WandbLogger(
            project  = cfg.WANDB_PROJECT,
            entity   = cfg.WANDB_ENTITY,
            name     = cfg.WANDB_NAME,
            save_dir = cfg.WANDB_SAVE_DIR,
            reinit = True,
        )

        # 3) 모델·데이터모듈 생성
        model = SimpleClassifier(
            model_name      = cfg.MODEL_NAME,
            num_classes     = cfg.NUM_CLASSES,
            optimizer_params= cfg.OPTIMIZER_PARAMS,
            scheduler_params= cfg.SCHEDULER_PARAMS,
        )
        datamodule = TinyImageNetDatasetModule(batch_size=cfg.BATCH_SIZE)

        # 4) Trainer 인스턴스
        trainer = Trainer(
            accelerator           = cfg.ACCELERATOR,
            devices               = cfg.DEVICES,
            precision             = cfg.PRECISION_STR,
            max_epochs            = cfg.NUM_EPOCHS,
            check_val_every_n_epoch = cfg.VAL_EVERY_N_EPOCH,
            logger                = wandb_logger,
            callbacks             = [
                LearningRateMonitor(logging_interval='epoch'),
                ModelCheckpoint(save_top_k=1, monitor='accuracy/val', mode='max'),
            ],
        )

        # 5) 학습 및 검증
        print(f"\n\n===== Starting Experiment {i+1}/{n}: "
              f"{cfg.MODEL_NAME}, OPT={cfg.OPTIMIZER_PARAMS}, SCHED={cfg.SCHEDULER_PARAMS} =====\n")
        trainer.fit(model, datamodule=datamodule)
        trainer.validate(ckpt_path='best', datamodule=datamodule)

        # 6) GPU 메모리 정리 (필요 시)
        import torch; torch.cuda.empty_cache()

        wandb.finish()

if __name__ == "__main__":
    main()
