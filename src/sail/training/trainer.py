from dataclasses import dataclass
import torch, tqdm, os
from typing import Any
from ..core.base_model import BaseModelWrapper
from ..core.base_dataset import BaseDatasetAdapter

@dataclass
class TrainConfig:
    epochs: int = 3
    # lr: float = 3e-4
    lr: float = 0.00001
    weight_decay: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    ckpt_dir: str = "artifacts/checkpoints"
    ckpt_name: str = f"model.torch"

class Trainer:
    
    def __init__(self, model_wrapper: BaseModelWrapper, dataset_adapter: BaseDatasetAdapter, cfg: TrainConfig, model_name, batch_size):
        self.mw = model_wrapper
        self.ds = dataset_adapter
        self.cfg = cfg
        self.net = self.mw.build().to(cfg.device)
        # self.opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
        self.model_name = model_name
        self.batch_size = batch_size

    

    def fit(self) -> None:

        path = os.path.join(self.cfg.ckpt_dir)#, self.cfg.ckpt_name)
        
        self.net.train()

        best_loss = 1000000000000
        for ep in range(self.cfg.epochs):


            with open(f"{path}/records.txt", "a") as f:
                f.write('Epoch {}/{}\n'.format(ep, self.cfg.epochs - 1))

            with open(f"{path}/records.txt", "a") as f:
                f.write('----------\n')
            

            # training loop
            self.mw.net.train()
            pbar = tqdm.tqdm(self.ds.train_loader(), desc=f"epoch {ep+1}/{self.cfg.epochs}")
            running_train_loss = 0
            for c, batch in enumerate(pbar):
                
                batch = {k: (v.to(self.cfg.device) if hasattr(v, "to") else v) for k,v in batch.items()}
                # with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                pred = self.mw.forward(batch)                
                loss = self.mw.compute_loss(pred, batch)
                loss.backward()

                if self.model_name == "geoconv":
                    if (c % self.batch_size == 0) & (c != 0):
                        self.opt.step()
                        self.opt.zero_grad(set_to_none=True)
                else:
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)
                
                # self.scaler.scale(loss).backward()
                # self.scaler.step(self.opt)
                # self.scaler.update()
                pbar.set_postfix(loss=float(loss))
                running_train_loss += loss.item()
        

            # validation loop
            self.mw.net.eval()
            pbar = tqdm.tqdm(self.ds.val_loader(), desc=f"epoch {ep+1}/{self.cfg.epochs}")
            running_val_loss = 0
            for batch in pbar:
                batch = {k: (v.to(self.cfg.device) if hasattr(v, "to") else v) for k,v in batch.items()}
                with torch.no_grad():
                    pred = self.mw.forward(batch)
                    loss = self.mw.compute_loss(pred, batch)
                pbar.set_postfix(loss=float(loss))
                running_val_loss += loss.item()

            with open(f"{path}/records.txt", "a") as f:
                f.write('{} Loss: {:.4f}\n'.format("Training", running_train_loss / len(self.ds.train_loader())))
                f.write('{} Loss: {:.4f}\n'.format("Validation", running_val_loss / len(self.ds.val_loader())))


            val_loss = running_val_loss / len(self.ds.val_loader())
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = self.mw.net.state_dict()
                self.mw.save(path = os.path.join(self.cfg.ckpt_dir, f"model_epoch{ep}.torch"))
        

        # os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        # self.mw.save(path)
        print(f"Saved checkpoint to {path}")






