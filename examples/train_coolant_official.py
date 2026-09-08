#!/usr/bin/env python3
"""
Official COOLANT Training Script (ACM MM '23, arXiv:2302.14057)

Loss schedule per paper §3.2:
  Task 1a — Consistency:   L_ITM  = CosineEmbeddingLoss(e_s_t, e_s_v, ±1)   (§3.2.1)
  Task 1b — Contrastive:   L_ITC  = symmetric InfoNCE(m_t, m_v)               (§3.2.2)
  Task 1c — Soft distill:  L_SEM  = soft-CE(S_ITM → P_ITC)                   (§3.2.4)
            Combined:       L_CL   = L_ITC + λ·L_SEM                          (Eq. 7)
  Task 2  — Detection:     L_DET  = L_CE + 0.5·L_KL                          (§3.4.3)

GatedMLP (SwiGLU) replaces all standard MLPs — the only modification vs paper.

Based on: https://github.com/wishever/COOLANT/blob/main/twitter/twitter.py
"""

import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import torch.nn.functional as F
import random
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

# Import our modules
from src.models.coolant_official import COOLANT_Official
from src.processing.simple_dataloader import create_dataloaders

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_data(
    text: torch.Tensor, image: torch.Tensor, label: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare data for similarity learning following official repository.

    Creates matched and unmatched pairs for contrastive learning.
    Handles edge case when no real news (label == 0) exists in batch.
    """
    nr_index = [i for i, l in enumerate(label) if l == 0]  # Non-rumor (real news)

    # Handle edge case: no real news in batch
    if len(nr_index) == 0:
        # Use all samples if no real news available
        nr_index = list(range(len(label)))

    # Ensure at least 2 samples for pair creation
    if len(nr_index) < 2:
        # Duplicate existing samples to create pairs
        nr_index = nr_index * 2 if len(nr_index) == 1 else [0, 0]

    text_nr = text[nr_index]
    image_nr = image[nr_index]
    fixed_text = copy.deepcopy(text_nr)
    matched_image = copy.deepcopy(image_nr)
    unmatched_image = copy.deepcopy(image_nr).roll(shifts=3, dims=0)

    return fixed_text, matched_image, unmatched_image


class COOLANTTrainer:
    """Trainer for official COOLANT implementation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(
            config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        )

        # Set random seeds
        seed = config.get("seed", 825)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Initialize model
        self.model = COOLANT_Official(config).to(self.device)

        # Optimizers — one per module (paper approach)
        self.optim_similarity = torch.optim.Adam(
            self.model.similarity_module.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("l2", 0),
        )

        if self.model.use_itc:
            # CLIPModule trained jointly with L_ITC + λ·L_SEM
            self.optim_itc = torch.optim.AdamW(
                self.model.clip_module.parameters(), lr=0.001, weight_decay=5e-4
            )

        self.optim_detection = torch.optim.Adam(
            self.model.detection_module.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("l2", 0),
        )

        # Loss functions — paper §3.2 notation
        # L_ITM handled via model.compute_loss_itm()
        # L_ITC handled via model.compute_loss_itc()
        # L_SEM handled via model.compute_loss_sem()
        self.loss_func_ce = torch.nn.CrossEntropyLoss()  # for L_CE in L_DET
        self.loss_func_kl = torch.nn.KLDivLoss(
            reduction="batchmean"
        )  # for L_KL in L_DET

        # Training state
        self.best_acc = 0
        self.step = 0

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch following official repository approach."""
        self.model.similarity_module.train()
        if self.model.use_itc:
            self.model.clip_module.train()
        self.model.detection_module.train()

        # Metrics tracking
        corrects_similarity = 0
        corrects_detection = 0
        loss_itm_total = 0  # L_ITM (§3.2.1)
        loss_itc_total = 0  # L_ITC (§3.2.2)
        loss_sem_total = 0  # L_SEM (§3.2.4, soft distillation)
        loss_det_total = 0  # L_DET (§3.4.3)
        similarity_count = 0
        detection_count = 0

        for i, (batch) in tqdm(enumerate(train_loader), desc="Training"):
            # Extract batch data
            text = batch["text_features"].to(self.device)
            image = batch["image_features"].to(self.device)
            label = batch["labels"].to(self.device)

            batch_size = text.shape[0]

            # Prepare data for similarity learning
            fixed_text, matched_image, unmatched_image = prepare_data(
                text, image, label
            )
            fixed_text = fixed_text.to(self.device)
            matched_image = matched_image.to(self.device)
            unmatched_image = unmatched_image.to(self.device)

            # ── TASK 1a: Consistency Learning (L_ITM, §3.2.1) ─────────────────
            # SimilarityModule → shared embeddings e_s^t, e_s^v
            e_s_t_m, e_s_v_m, pred_sim_m = self.model.similarity_module(
                fixed_text, matched_image
            )
            e_s_t_u, e_s_v_u, pred_sim_u = self.model.similarity_module(
                fixed_text, unmatched_image
            )

            similarity_pred = torch.cat(
                [pred_sim_m.argmax(1), pred_sim_u.argmax(1)], dim=0
            )
            # classifier labels: 1=matched, 0=unmatched (for accuracy)
            sim_cls_lbl = torch.cat(
                [
                    torch.ones(pred_sim_m.shape[0]),
                    torch.zeros(pred_sim_u.shape[0]),
                ]
            ).to(self.device)
            # cosine loss labels: +1=matched, -1=unmatched
            sim_cos_lbl = torch.cat(
                [
                    torch.ones(e_s_t_m.shape[0]),
                    -torch.ones(e_s_t_u.shape[0]),
                ]
            ).to(self.device)

            e_s_t_all = torch.cat([e_s_t_m, e_s_t_u], dim=0)
            e_s_v_all = torch.cat([e_s_v_m, e_s_v_u], dim=0)

            l_itm = self.model.compute_loss_itm(e_s_t_all, e_s_v_all, sim_cos_lbl)

            self.optim_similarity.zero_grad()
            l_itm.backward()
            self.optim_similarity.step()

            corrects_similarity += similarity_pred.eq(sim_cls_lbl).sum().item()
            loss_itm_total += l_itm.item() * e_s_t_all.shape[0]

            # ── TASK 1b+c: Contrastive + Soft Distillation (L_ITC + λ·L_SEM, §3.2.2+4) ──
            if self.model.use_itc:
                # Forward CLIPModule → m^t, m^v
                m_t_itc, m_v_itc = self.model.clip_module(text, image)

                # L_ITC: symmetric InfoNCE (§3.2.2)
                l_itc = self.model.compute_loss_itc(m_t_itc, m_v_itc)

                # L_SEM: soft distillation from SimilarityModule → CLIPModule (§3.2.4)
                # Need e_s^t, e_s^v for this batch (full batch, not pair-subset)
                with torch.no_grad():
                    e_s_t_full, e_s_v_full, _ = self.model.similarity_module(
                        text, image
                    )
                l_sem = self.model.compute_loss_sem(
                    m_t_itc, m_v_itc, e_s_t_full, e_s_v_full
                )

                # L_CL = L_ITC + λ·L_SEM (Eq. 7)
                l_cl = l_itc + self.model.sem_weight * l_sem

                self.optim_itc.zero_grad()
                l_cl.backward()
                self.optim_itc.step()

                loss_itc_total += l_itc.item() * text.shape[0]
                loss_sem_total += l_sem.item() * text.shape[0]

            # ── TASK 2: Detection (L_DET = L_CE + 0.5·L_KL, §3.4.3) ────────────
            # m^t, m^v for CrossModule — detached: Task 2 chỉ train DetectionModule
            with torch.no_grad():
                if self.model.use_itc:
                    m_t, m_v = self.model.clip_module(text, image)
                else:
                    m_t, m_v, _ = self.model.similarity_module(text, image)

            detection_logits, attention_score, skl_score = self.model.detection_module(
                text, image, m_t, m_v
            )

            l_ce = self.loss_func_ce(detection_logits, label)
            l_kl = self.loss_func_kl(
                F.log_softmax(attention_score.clamp(-10, 10), dim=1),
                F.softmax(skl_score.clamp(-10, 10), dim=1),
            )
            l_det = l_ce + 0.5 * l_kl

            self.optim_detection.zero_grad()
            l_det.backward()
            self.optim_detection.step()

            pre_label_detection = detection_logits.argmax(1)
            corrects_detection += (
                pre_label_detection.eq(label.view_as(pre_label_detection)).sum().item()
            )
            loss_det_total += l_det.item() * text.shape[0]
            detection_count += text.shape[0]
            similarity_count += e_s_t_all.shape[0]
            self.step += 1

        metrics = {
            "loss_itm": loss_itm_total / max(similarity_count, 1),
            "loss_itc": loss_itc_total / max(detection_count, 1),
            "loss_sem": loss_sem_total / max(detection_count, 1),
            "loss_det": loss_det_total / max(detection_count, 1),
            "acc_detection": corrects_detection / max(detection_count, 1),
            "acc_similarity": corrects_similarity / max(similarity_count, 1),
        }
        return metrics

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model following official repository approach."""
        self.model.similarity_module.eval()
        if self.model.use_itc:
            self.model.clip_module.eval()
        self.model.detection_module.eval()

        detection_count = 0
        loss_detection_total = 0
        detection_labels = []
        detection_predictions = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                text = batch["text_features"].to(self.device)
                image = batch["image_features"].to(self.device)
                label = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(text, image)
                detection_logits = outputs["detection_logits"]

                # Compute L_CE for tracking
                loss_detection = self.loss_func_ce(detection_logits, label)
                loss_detection_total += loss_detection.item() * text.shape[0]
                detection_count += text.shape[0]

                # Collect predictions
                pre_label_detection = detection_logits.argmax(1)
                detection_labels.extend(label.cpu().numpy())
                detection_predictions.extend(pre_label_detection.cpu().numpy())

        # Compute metrics
        loss_detection_test = loss_detection_total / detection_count
        acc_detection_test = accuracy_score(detection_labels, detection_predictions)
        cm_detection = confusion_matrix(detection_labels, detection_predictions)
        cr_detection = classification_report(detection_labels, detection_predictions)

        return {
            "loss_detection": loss_detection_test,
            "acc_detection": acc_detection_test,
            "confusion_matrix": cm_detection,
            "classification_report": cr_detection,
        }

    def train(
        self, train_loader: DataLoader, test_loader: DataLoader, num_epochs: int = 50
    ):
        """Main training loop following official repository."""
        logger.info("Starting COOLANT training...")

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)

            # Evaluate
            test_metrics = self.evaluate(test_loader)

            # Print results (following official format)
            logger.info(f"--- TASK1 Consistency (L_ITM) + Contrastive (L_ITC) ---")
            logger.info(
                f"[Epoch: {epoch}] "
                f'L_ITM={train_metrics["loss_itm"]:.4f}  '
                f'L_ITC={train_metrics["loss_itc"]:.4f}  '
                f'L_SEM={train_metrics["loss_sem"]:.4f}'
            )

            logger.info(f"--- TASK2 Detection (L_DET) ---")
            if test_metrics["acc_detection"] > self.best_acc:
                self.best_acc = test_metrics["acc_detection"]
                logger.info(
                    f'New best accuracy! Classification Report:\n{test_metrics["classification_report"]}'
                )

                # Save best models
                save_dir = Path(self.config.get("save_dir", "./checkpoints"))
                save_dir.mkdir(exist_ok=True)

                torch.save(
                    self.model.similarity_module.state_dict(),
                    save_dir / "best_similarity_module.pth",
                )
                torch.save(
                    self.model.detection_module.state_dict(),
                    save_dir / "best_detection_model.pth",
                )
                logger.info(f"Saved best models to {save_dir}")

            logger.info(
                f"EPOCH = {epoch + 1}\n"
                f"acc_detection_train = {train_metrics['acc_detection']:.3f}\n"
                f"acc_detection_test = {test_metrics['acc_detection']:.3f}\n"
                f"best_acc = {self.best_acc:.3f}\n"
                f"loss_det_train = {train_metrics['loss_det']:.3f}\n"
                f"loss_det_test = {test_metrics['loss_detection']:.3f}\n"
            )

            logger.info(
                f'--- TASK2 Detection Confusion Matrix ---\n{test_metrics["confusion_matrix"]}\n'
            )

        logger.info("Training completed!")


def main():
    """Main training function."""
    # Configuration
    config = {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "batch_size": 64,
        "lr": 1e-3,
        "l2": 0,
        "num_epochs": 50,
        "seed": 825,
        "save_dir": "./checkpoints",
        # Model configuration
        "shared_dim": 128,
        "sim_dim": 64,
        "feature_dim": 96,  # 64 + 16 + 16
        "h_dim": 64,
        # Data configuration
        "json_path": "src/data/json/news_data.json",
        "image_base_dir": "src/data/jpg",
        "preprocessed_dir": "preprocessed_coolant",
        "label_mapping": {"thanh_nien": 0, "dan_tri": 1, "vnexpress": 2, "tuoitre": 3},
    }

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        json_path=config["json_path"],
        image_base_dir=config["image_base_dir"],
        batch_size=config["batch_size"],
        label_mapping=config["label_mapping"],
        preprocessed_dir=config["preprocessed_dir"],
    )

    # Initialize trainer
    trainer = COOLANTTrainer(config)

    # Start training
    trainer.train(train_loader, test_loader, config["num_epochs"])


if __name__ == "__main__":
    main()
