import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np


def train_reconstruct(args, cfg, model, train_loader, val_loader, logger, model_save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss(reduction='mean')

    best_val_loss = 1e10
    best_val_epoch = 0
    patience_count = 0
    for epoch in tqdm(range(args.epoch), desc='Training'):
        train_epoch_loss = 0
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            x = batch[0].to(args.device)
            x_hat = model(x, teacher_forcing_ratio=cfg['teacher_forcing_ratio'])
            loss = criterion(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()
        train_epoch_loss = train_epoch_loss / len(train_loader)
        
        val_epoch_loss = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                x = batch[0].to(args.device)
                x_hat = model(x, teacher_forcing_ratio=0)

                loss = criterion(x_hat, x)
                val_epoch_loss += loss.item()
            val_epoch_loss = val_epoch_loss / len(val_loader)

        logger.info(f"Epoch {epoch+1} | Train loss {train_epoch_loss:.5f} / Validation loss {val_epoch_loss:.5f}")
        if val_epoch_loss < best_val_loss:
            logger.info(f"Lower validation loss, saving the model.. Current {best_val_loss:.5f} → {val_epoch_loss:.5f}")
            best_val_loss = val_epoch_loss
            best_val_epoch = epoch+1
            patience_count = 0
            torch.save(model, model_save_path)
        else:
            patience_count += 1
            if patience_count == args.patience:
                logger.info("Early stop triggered..")
                break
    logger.info(f"Best model saved to {model_save_path} : Best epoch {best_val_epoch}, Best val loss {best_val_loss:.5f}")


## Calculate the mean and var of modeling error using validation set
def normalize_anomaly_score_reconstruct(args, best_model, val_loader):
    val_error = []
    best_model.eval()
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Error adjusting"):
            x = batch[0].to(args.device)
            x_hat = best_model(x, teacher_forcing_ratio=0)
            loss = F.l1_loss(x_hat, x, reduction='none')
            loss = torch.mean(loss, dim=(1)).detach().cpu().numpy()
            val_error.append(loss)
    val_error = np.concatenate(val_error, axis=0)
    mean_error = np.mean(val_error, axis=0)
    std_error = np.cov(val_error.T)

    return mean_error, std_error


def test_reconstruct(args, best_model, test_loader, AS_calculator):
    test_anomaly_score = []
    test_label = []
    best_model.eval()
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing"):
            x = batch[0].to(args.device) # [B,w,N]
            label = batch[1] # [B,w]
            x_hat = best_model(x, teacher_forcing_ratio=0) # [B,w,N]

            loss = F.l1_loss(x_hat, x, reduction='none')
            loss = torch.mean(loss, dim=(1)).detach().cpu().numpy()
            anomaly_score = AS_calculator(loss) # (B,)
            
            test_anomaly_score.append(anomaly_score)
            test_label.append(label[:,-1])

    test_anomaly_score = np.concatenate(test_anomaly_score, axis=0)
    test_label = np.concatenate(test_label, axis=0)

    return test_anomaly_score, test_label

