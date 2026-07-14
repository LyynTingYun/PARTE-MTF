from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, parse_setting, write_result_csv
from utils.metrics import metric
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from collections import OrderedDict
from tensorboardX import SummaryWriter
import random
from tqdm import tqdm

warnings.filterwarnings('ignore')


class Exp_PARTE(Exp_Basic):
    def __init__(self, args):
        super(Exp_PARTE, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.task_name == "finetune":
            
            if self.args.task_name == "finetune":
                model = self._load_pretrain_checkpoint(model)

        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _load_pretrain_checkpoint(self, model):
        ckpt_path = os.path.join(
            self.args.pretrain_checkpoints,
            self.args.data,
            "ckpt_best.pth"
        )
    
        assert os.path.exists(ckpt_path), \
            f"Pretrained checkpoint not found: {ckpt_path}"
    
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]
    
        # remove prediction head parameters
        state_dict.pop("pred_head.linear.weight", None)
        state_dict.pop("pred_head.linear.bias", None)
    
        msg = model.load_state_dict(state_dict, strict=False)
    
        print(f">>>>> Loaded pretrained checkpoint from {ckpt_path}")
        print(f"Missing keys: {msg.missing_keys}")
        print(f"Unexpected keys: {msg.unexpected_keys}")
    
        return model

    def _save_pretrain_checkpoint(self, path, epoch, best=False):
        
            if best:
                ckpt_path = os.path.join(path, "ckpt_best.pth")
            else:
                ckpt_path = os.path.join(path, f"ckpt{epoch + 1}.pth")
        
            torch.save({"epoch": epoch,
                        "model_state_dict": self.model.state_dict()
                       },
                       ckpt_path
                      )
            return ckpt_path

    def pretrain(self):
            
        # data preparation
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.pretrain_checkpoints, self.args.data)
        if not os.path.exists(path):
            os.makedirs(path)

        # optimizer
        model_optim = self._select_optimizer()
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optim, T_max=self.args.train_epochs)
        # pre-training
        min_vali_loss = None
        for epoch in range(self.args.train_epochs):
            
            start_time = time.time()
            print(
                    f"\n{'='*20} "
                    f"Pretrain Stage, Now is Epoch [{epoch+1}/{self.args.train_epochs}] "
                    f"{'='*20}"
                )
            train_loss, train_mse_loss, train_topo_loss = self._pretrain_one_epoch(train_loader, model_optim, model_scheduler)
            print(
                    f"{'-'*20} "
                    f"Validation of Pretrain Stage on Epoch [{epoch+1}/{self.args.train_epochs}] "
                    f"{'-'*20}"
                )
            vali_loss, vali_mse_loss, vali_topo_loss = self._valid_one_epoch(vali_loader)
            
            # log and Loss
            end_time = time.time()
       
            print(
                "Epoch: {0}, Lr: {1:.7f}, Time: {2:.2f}s | Train Loss: {3:.4f} ({4:.3f}, {5:.3f}) Val Loss: {6:.4f} ({7:.3f}, {8:.3f})"
                .format(
                    epoch,
                    model_scheduler.get_last_lr()[0],
                    end_time - start_time,
                    train_loss,
                    train_mse_loss,
                    train_topo_loss,
                    vali_loss,
                    vali_mse_loss,
                    vali_topo_loss
                )
            )

            loss_scalar_dict = {
                'train_loss': train_loss,
                'train_mse_loss' : train_mse_loss,
                'train_cont_loss' : train_topo_loss,
                'vali_loss': vali_loss,
                'vali_mse_loss': vali_mse_loss,
                'vali_cont_loss': vali_topo_loss
            }

            self.writer.add_scalars(f"/pretrain_loss", loss_scalar_dict, epoch)

            # checkpoint saving
            # save best model
            if min_vali_loss is None or vali_loss < min_vali_loss:
            
                print(
                    "Validation loss decreased ({0:.4f} --> {1:.4f}). Saving best model ..."
                    .format(
                        min_vali_loss if min_vali_loss is not None else float("inf"),
                        vali_loss
                    )
                )
            
                min_vali_loss = vali_loss
                self._save_pretrain_checkpoint(path=path, epoch=epoch, best=True)
            
            # save every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Saving model at epoch {epoch + 1}...")
                self._save_pretrain_checkpoint(path=path, epoch=epoch, best=False)


    def _pretrain_one_epoch(self, train_loader, model_optim, model_scheduler):
        
        train_loss = []
        train_topo_loss = []
        train_mse_loss = []
        criterion = self._select_criterion()
        self.model.train()
        for i, batch in enumerate(tqdm(train_loader)):

            batch_x, batch_y, pd_tensor, pd_mask, batch_x_mark, batch_y_mark = batch['ts'], batch['target'], batch['pd_tensor'], batch['pd_mask'], batch['x_mark'], batch['y_mark']
            pd_tensor = pd_tensor.float().to(self.device)
            pd_mask = pd_mask.int().to(self.device)
            
            model_optim.zero_grad()
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)

            outputs, topo_loss = self.model(batch_x, batch_x_mark, pd_tensor, pd_mask)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

            mse_loss = criterion(outputs, batch_y) 
            loss = mse_loss + self.args.alpha * topo_loss
            
            # backward
            loss.backward()
            model_optim.step()
            
            # record
            train_loss.append(loss.item())
            train_mse_loss.append(mse_loss.item())
            train_topo_loss.append(topo_loss.item())

        model_scheduler.step()
        
        train_loss = np.average(train_loss)
        train_mse_loss = np.average(train_mse_loss)
        train_topo_loss = np.average(train_topo_loss)

        return train_loss, train_mse_loss, train_topo_loss

    def _valid_one_epoch(self, vali_loader):
        valid_loss = []
        valid_topo_loss = []
        valid_mse_loss = []
        criterion = self._select_criterion()
        self.model.eval()

        for i, batch in enumerate(tqdm(vali_loader)):

            batch_x, batch_y, pd_tensor, pd_mask, batch_x_mark, batch_y_mark = batch['ts'], batch['target'], batch['pd_tensor'], batch['pd_mask'], batch['x_mark'], batch['y_mark']
            pd_tensor = pd_tensor.float().to(self.device)
            pd_mask = pd_mask.int().to(self.device)
            
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)

            outputs, topo_loss = self.model(batch_x, batch_x_mark, pd_tensor, pd_mask)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

            mse_loss = criterion(outputs, batch_y) 
            loss = mse_loss + self.args.alpha * topo_loss
            
            # Record
            valid_loss.append(loss.item())
            valid_mse_loss.append(mse_loss.item())
            valid_topo_loss.append(topo_loss.item())

        valid_loss = np.average(valid_loss)
        valid_mse_loss = np.average(valid_mse_loss)
        valid_topo_loss = np.average(valid_topo_loss)

        self.model.train()
        return valid_loss, valid_mse_loss, valid_topo_loss
    
    def train(self, setting):

        # data preparation
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # Optimizer
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)    
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            start_time = time.time()
            print(
                    f"\n{'='*20} "
                    f"Finetune Stage, Now is Epoch [{epoch+1}/{self.args.train_epochs}] "
                    f"{'='*20}"
                )
            for i, batch in enumerate(tqdm(train_loader)):
                
                batch_x, batch_y, pd_tensor, pd_mask, batch_x_mark, batch_y_mark = batch['ts'], batch['target'], batch['pd_tensor'], batch['pd_mask'], batch['x_mark'], batch['y_mark']
                pd_tensor = pd_tensor.float().to(self.device)
                pd_mask = pd_mask.int().to(self.device)
                iter_count += 1
                model_optim.zero_grad()

                # to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # encoder
                outputs = self.model(batch_x, batch_x_mark, pd_tensor, pd_mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # loss
                loss = criterion(outputs, batch_y)
                loss.backward()
                model_optim.step()

                # record
                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)

            end_time = time.time()
            print(
            "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f}".format(
                epoch + 1, train_steps, end_time - start_time, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        self.lr = model_optim.param_groups[0]['lr']

        return self.model

    def vali(self, vali_loader, criterion):
        total_loss = []

        self.model.eval()
        print(
                f"{'-'*20} "
                f"Validation of Finetune Stage"
                f"{'-'*20}"
             )
        with torch.no_grad():
            for i, batch in enumerate(tqdm(vali_loader)):
                
                batch_x, batch_y, pd_tensor, pd_mask, batch_x_mark, batch_y_mark = batch['ts'], batch['target'], batch['pd_tensor'], batch['pd_mask'], batch['x_mark'], batch['y_mark']
                pd_tensor = pd_tensor.float().to(self.device)
                pd_mask = pd_mask.int().to(self.device)

                # to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # encoder
                outputs = self.model(batch_x, batch_x_mark, pd_tensor, pd_mask)

                # loss
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)

                # record
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        folder_path = './outputs/test_results/'
        dataset_folder = os.path.join(folder_path, self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)

        self.model.eval()

        print(
                f"{'-'*20} "
                f"Test of Finetune Stage"
                f"{'-'*20}"
             )
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                
                batch_x, batch_y, pd_tensor, pd_mask, batch_x_mark, batch_y_mark = batch['ts'], batch['target'], batch['pd_tensor'], batch['pd_mask'], batch['x_mark'], batch['y_mark']
                pd_tensor = pd_tensor.float().to(self.device)
                pd_mask = pd_mask.int().to(self.device)

                # to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # encoder
                outputs = self.model(batch_x, batch_x_mark, pd_tensor, pd_mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

                if i % 2000 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(dataset_folder, str(i) + f'{self.args.pred_len}.png'))

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('{0}->{1}, mse:{2:.3f}, mae:{3:.3f}'.format(self.args.seq_len, self.args.pred_len, mse, mae))
        write_result_csv(folder_path + "results.csv", setting, mse, mae)