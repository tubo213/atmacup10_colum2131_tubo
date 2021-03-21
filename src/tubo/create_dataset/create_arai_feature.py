import os
import random
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
import umap
from pathlib import Path
from albumentations.pytorch import ToTensorV2
from catalyst.dl import SupervisedRunner, Runner
from catalyst.core import Callback, CallbackOrder, IRunner
from sklearn.model_selection import KFold
from tqdm import tqdm

#araiさんの自己なんとかかんとか学習の特徴量作成するファイル、cv無し、最終層のReluを消した
def _create_random_image(sample: pd.DataFrame) -> np.ndarray:
    """
    配置はランダムで色の比率がsampleで指示された値になるようにした
    10x10の画像を生成する
    """
    # まず一次元で定義しておく
    image = np.zeros((100, 3), dtype=np.uint8)
    # sampleの頭から1行ずつその行の色をその行のratio_int分だけコピーして画像を埋める
    head = 0
    for i, row in sample.iterrows():
        # sampleの行に書かれた色
        patch = np.array([[row.color_r, row.color_g, row.color_b]], dtype=np.uint8)
        # sampleの行に書かれたratio_int分だけコピーする
        patch = np.tile(patch, row.ratio_int).reshape(row.ratio_int, -1)
        # 画像を上の手順で出した色で埋める
        image[head:head + row.ratio_int, :] = patch
        head += row.ratio_int
    # 乱数で順番をランダム化する
    indices = np.random.permutation(np.arange(100))
    image = image[indices, :].reshape(10, 10, 3)
    return image

class ColorImageDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, transforms=None):
        self.object_id = df["object_id"].unique()
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.object_id)

    def __getitem__(self, idx: int):
        object_id = self.object_id[idx]
        sample = self.df.query(f"object_id == '{object_id}'")[
            ["ratio_int", "color_r", "color_g", "color_b"]]
        # 負例のサンプリングを行う
        while True:
            neg_sample_id = np.random.choice(self.object_id)
            if neg_sample_id != object_id:
                break
        neg_sample = self.df.query(f"object_id == '{neg_sample_id}'")[
            ["ratio_int", "color_r", "color_g", "color_b"]]

        # アンカー画像の生成
        anchor = _create_random_image(sample)
        # 正例の生成
        pos = _create_random_image(sample)
        # 負例の生成
        neg = _create_random_image(neg_sample)

        anchor = self.transforms(image=anchor)["image"]
        pos = self.transforms(image=pos)["image"]
        neg = self.transforms(image=neg)["image"]
        return anchor, pos, neg
    
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3))

    def forward(self, x):
        return self.cnn_encoder(x).mean(dim=[2, 3])

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity()

    def forward(self, anchor, pos, neg):
        pos_loss = 1.0 - self.cos(anchor, pos).mean(dim=0)
        neg_loss = self.cos(anchor, neg).mean(dim=0)
        return pos_loss + neg_loss

class ContrastRunner(Runner):
    def predict_batch(self, batch, **kwargs):
        return super().predict_batch(batch, **kwargs)

    def _handle_batch(self, batch):
        anchor, pos, neg = batch[0], batch[1], batch[2]
        anchor = anchor.to(self.device)
        pos = pos.to(self.device)
        neg = neg.to(self.device)

        anchor_emb = self.model(anchor)
        pos_emb = self.model(pos)
        neg_emb = self.model(neg)

        loss = self.criterion(anchor_emb, pos_emb, neg_emb)
        self.batch_metrics.update({
            "loss": loss
        })

        self.input = batch
        if self.is_train_loader:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

class SchedulerCallback(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.Scheduler)

    def on_loader_end(self, state: IRunner):
        lr = state.scheduler.get_last_lr()
        state.epoch_metrics["lr"] = lr[0]
        if state.is_train_loader:
            state.scheduler.step()
            
def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

INPUT_PATH = '../../../data/'
OUTDIR = Path("../PaletteEmbedding")
OUTDIR.mkdir(exist_ok=True, parents=True)
OUTPUT_PATH = '../../../features/tubo/'

def main():
    sns.set_context("talk")
    plt.style.use("ggplot")
    palette = pd.read_csv(INPUT_PATH + 'palette.csv')
    
    palette["ratio_int"] = palette["ratio"].map(lambda x: int(np.round(100 * x)))
    palette_group_dfs = []
    for _, df in tqdm(palette.groupby("object_id"),
                    total=palette["object_id"].nunique()):
        # 足し合わせた和が100を超過する場合
        if df["ratio_int"].sum() > 100:
            n_excess = df["ratio_int"].sum() - 100
            # ちょっと雑だが一番比率が多い色の割合を減らすことで和を100に揃える
            max_ratio_int_idx = df["ratio_int"].idxmax()
            df.loc[max_ratio_int_idx, "ratio_int"] -= n_excess
        elif df["ratio_int"].sum() < 100:
            n_lack = 100 - df["ratio_int"].sum()
            max_ratio_int_idx = df["ratio_int"].idxmax()
            df.loc[max_ratio_int_idx, "ratio_int"] += n_lack
        else:
            pass
        palette_group_dfs.append(df)

    new_palette = pd.concat(palette_group_dfs, axis=0).reset_index(drop=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(1213)
    transforms = A.Compose([A.Normalize(),ToTensorV2()])
    
    dataset = ColorImageDataset(new_palette,transforms)

    loader = torchdata.DataLoader(
        dataset,batch_size=1024,shuffle=True,num_workers=16
    )

    model = CNNModel().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    callbacks = [SchedulerCallback()]
    runner = ContrastRunner(device=device)
    runner.train(model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    callbacks=callbacks,
                    loaders={"train": loader},
                    num_epochs=30,
                    logdir=OUTDIR/ 'model_log',
                    verbose=True)
    
    embeddings = []
    object_ids = []
    dataset = ColorImageDataset(new_palette,transforms)
    object_ids.extend(dataset.object_id.tolist())

    loader = torchdata.DataLoader(dataset,batch_size=1024,shuffle=False,num_workers=16)
    model = CNNModel()
    ckpt = torch.load(OUTDIR / 'model_log/checkpoints/best.pth')
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    for anchor, _, _ in tqdm(loader):
        anchor = anchor.to(device)
        with torch.no_grad():
            embedding = model(anchor).detach().cpu().numpy()
        embeddings.append(embedding)
    
    all_embeddings = np.concatenate(embeddings, axis=0)
    embedding_df = pd.DataFrame(all_embeddings, 
                            columns=[f"color_embedding_{i}" for i in range(len(all_embeddings[0]))],
                            index=object_ids)
    
    embedding_df.to_pickle(OUTPUT_PATH + 'arai_feature.pickle')

if __name__ == '__main__':
    main()
    

    
    
    
    
    