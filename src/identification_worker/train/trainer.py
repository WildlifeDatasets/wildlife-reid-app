import numpy as np
import torch
from tqdm import tqdm
from wildlife_tools.train import BasicTrainer


class CarnivoreIDTrainer(BasicTrainer):
    def __init__(self, val_dataset, start_epoch: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.val_dataset = val_dataset
        self.start_epoch = start_epoch
        self.epoch = start_epoch

    def train(self):
        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        for e in range(self.start_epoch, self.epochs):
            epoch_data = self.train_epoch(loader)
            val_epoch_data = self.val_epoch(val_loader)
            epoch_data.update(val_epoch_data)

            if self.epoch_callback:
                self.epoch_callback(trainer=self, epoch_data=epoch_data)

            self.epoch += 1

    def train_epoch(self, loader):
        model = self.model.train()
        losses = []
        for i, batch in enumerate(tqdm(loader, desc=f"Epoch {self.epoch}: ", mininterval=1)):
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)

            out = model(x)
            loss = self.objective(out, y)
            loss.backward()
            if (i - 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            losses.append(loss.detach().cpu())

        if self.scheduler:
            self.scheduler.step()

        return {"train_loss_epoch_avg": np.mean(losses)}

    def val_epoch(self, loader):
        model = self.model.eval()
        losses = []
        for i, batch in enumerate(tqdm(loader, desc=f"Epoch {self.epoch}: ", mininterval=1)):
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)

            out = model(x)
            loss = self.objective(out, y)
            losses.append(loss.detach().cpu())

        return {"val_loss_epoch_avg": np.mean(losses)}
