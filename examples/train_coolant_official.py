#!/usr/bin/env python3
"""
Official COOLANT Training Script

This script follows the official repository's training approach with:
- Separate tasks with individual optimizers
- Task 1: Similarity learning + CLIP contrastive learning
- Task 2: Detection with ambiguity learning
- Proper data preparation and loss computation

Based on: https://github.com/wishever/COOLANT/blob/main/twitter/twitter.py
"""

import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import math
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


def prepare_data(text: torch.Tensor, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare data for similarity learning following official repository.
    
    Creates matched and unmatched pairs for contrastive learning.
    """
    nr_index = [i for i, l in enumerate(label) if l == 0]  # Non-rumor (real news)
    if len(nr_index) < 2:
        nr_index.extend([np.random.randint(len(label)) for _ in range(2)])
    
    text_nr = text[nr_index]
    image_nr = image[nr_index]
    fixed_text = copy.deepcopy(text_nr)
    matched_image = copy.deepcopy(image_nr)
    unmatched_image = copy.deepcopy(image_nr).roll(shifts=3, dims=0)
    
    return fixed_text, matched_image, unmatched_image


def get_soft_label(label: torch.Tensor) -> torch.Tensor:
    """
    Generate soft labels for contrastive learning following official repository.
    """
    soft_label = []
    bs = len(label)
    for i, l in enumerate(label):
        if l == 0:  # Real news
            true_label = [0 for _ in range(bs)]
            true_label[i] = 1
            soft_label.append(true_label)
        else:  # Fake news
            soft_label.append([1./bs for _ in range(bs)])
    return torch.tensor(soft_label)


def soft_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Soft loss function from official repository.
    """
    logprobs = torch.nn.functional.log_softmax(input, dim=1)
    return -(target * logprobs).sum() / input.shape[0]


class COOLANTTrainer:
    """Trainer for official COOLANT implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu'))
        
        # Set random seeds
        seed = config.get('seed', 825)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Initialize model
        self.model = COOLANT_Official(config).to(self.device)
        
        # Initialize optimizers (separate for each task like official)
        self.optim_similarity = torch.optim.Adam(
            self.model.similarity_module.parameters(), 
            lr=config.get('lr', 1e-3), 
            weight_decay=config.get('l2', 0)
        )
        
        self.optim_clip = torch.optim.AdamW(
            self.model.clip_module.parameters(), 
            lr=0.001, 
            weight_decay=5e-4
        )
        
        self.optim_detection = torch.optim.Adam(
            self.model.detection_module.parameters(), 
            lr=config.get('lr', 1e-3), 
            weight_decay=config.get('l2', 0)
        )
        
        # Loss functions
        self.loss_func_similarity = torch.nn.CosineEmbeddingLoss(margin=0.2)
        self.loss_func_clip = torch.nn.CrossEntropyLoss()
        self.loss_func_detection = torch.nn.CrossEntropyLoss()
        self.loss_func_skl = torch.nn.KLDivLoss(reduction='batchmean')
        
        # Training state
        self.best_acc = 0
        self.step = 0
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch following official repository approach."""
        self.model.similarity_module.train()
        self.model.clip_module.train()
        self.model.detection_module.train()
        
        # Metrics tracking
        corrects_similarity = 0
        corrects_detection = 0
        loss_similarity_total = 0
        loss_clip_total = 0
        loss_detection_total = 0
        similarity_count = 0
        detection_count = 0
        
        for i, (batch) in tqdm(enumerate(train_loader), desc="Training"):
            # Extract batch data
            text = batch['text_features'].to(self.device)
            image = batch['image_features'].to(self.device)
            label = batch['labels'].to(self.device)
            
            batch_size = text.shape[0]
            
            # Prepare data for similarity learning
            fixed_text, matched_image, unmatched_image = prepare_data(text, image, label)
            fixed_text = fixed_text.to(self.device)
            matched_image = matched_image.to(self.device)
            unmatched_image = unmatched_image.to(self.device)
            
            # === TASK 1: Similarity Learning ===
            text_aligned_match, image_aligned_match, pred_similarity_match = self.model.similarity_module(fixed_text, matched_image)
            text_aligned_unmatch, image_aligned_unmatch, pred_similarity_unmatch = self.model.similarity_module(fixed_text, unmatched_image)
            
            # Prepare similarity labels
            similarity_pred = torch.cat([pred_similarity_match.argmax(1), pred_similarity_unmatch.argmax(1)], dim=0)
            similarity_label_0 = torch.cat([
                torch.ones(pred_similarity_match.shape[0]), 
                torch.zeros(pred_similarity_unmatch.shape[0])
            ], dim=0).to(self.device)
            similarity_label_1 = torch.cat([
                torch.ones(pred_similarity_match.shape[0]), 
                -1 * torch.ones(pred_similarity_unmatch.shape[0])
            ], dim=0).to(self.device)
            
            # Concatenate for similarity loss
            text_aligned_4_task1 = torch.cat([text_aligned_match, text_aligned_unmatch], dim=0)
            image_aligned_4_task1 = torch.cat([image_aligned_match, image_aligned_unmatch], dim=0)
            
            # Compute and backpropagate similarity loss
            loss_similarity = self.loss_func_similarity(text_aligned_4_task1, image_aligned_4_task1, similarity_label_1)
            
            self.optim_similarity.zero_grad()
            loss_similarity.backward()
            self.optim_similarity.step()
            
            corrects_similarity += similarity_pred.eq(similarity_label_0).sum().item()
            
            # === TASK 1: CLIP Contrastive Learning ===
            image_aligned, text_aligned = self.model.clip_module(image, text)
            logits = torch.matmul(image_aligned, text_aligned.T) * math.exp(0.07)
            labels = torch.arange(text.size(0)).to(self.device)
            
            # Get soft labels from similarity module
            text_sim, image_sim, _ = self.model.similarity_module(text, image)
            soft_label = torch.matmul(image_sim, text_sim.T) * math.exp(0.07)
            soft_label = soft_label.to(self.device)
            
            # Compute CLIP losses
            self.optim_clip.zero_grad()
            loss_clip_i = self.loss_func_clip(logits, labels)
            loss_clip_t = self.loss_func_clip(logits.T, labels)
            loss_clip = (loss_clip_i + loss_clip_t) / 2.
            
            image_loss = soft_loss(logits, F.softmax(soft_label, 1))
            caption_loss = soft_loss(logits.T, F.softmax(soft_label.T, 1))
            loss_soft = (image_loss + caption_loss) / 2.
            
            all_loss = loss_clip + 0.2 * loss_soft
            all_loss.backward()
            self.optim_clip.step()
            
            # === TASK 2: Detection ===
            # Use CLIP-aligned features for detection
            detection_logits, attention_score, skl_score = self.model.detection_module(
                text, image, text_aligned, image_aligned
            )
            
            # Compute detection loss
            loss_detection = self.loss_func_detection(detection_logits, label) + 0.5 * self.loss_func_skl(attention_score, skl_score)
            
            self.optim_detection.zero_grad()
            loss_detection.backward()
            self.optim_detection.step()
            
            # Update metrics
            pre_label_detection = detection_logits.argmax(1)
            corrects_detection += pre_label_detection.eq(label.view_as(pre_label_detection)).sum().item()
            
            loss_clip_total += loss_soft.item()
            loss_detection_total += loss_detection.item() * text.shape[0]
            similarity_count += (2 * fixed_text.shape[0] * 2)
            detection_count += text.shape[0]
            self.step += 1
        
        # Compute averages
        loss_detection_train = loss_detection_total / detection_count
        acc_detection_train = corrects_detection / detection_count
        acc_similarity_train = corrects_similarity / similarity_count
        
        return {
            'loss_detection': loss_detection_train,
            'acc_detection': acc_detection_train,
            'acc_similarity': acc_similarity_train,
            'loss_clip': loss_clip_total / self.step
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model following official repository approach."""
        self.model.similarity_module.eval()
        self.model.clip_module.eval()
        self.model.detection_module.eval()
        
        detection_count = 0
        loss_detection_total = 0
        detection_labels = []
        detection_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                text = batch['text_features'].to(self.device)
                image = batch['image_features'].to(self.device)
                label = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(text, image)
                detection_logits = outputs['detection_logits']
                
                # Compute detection loss
                loss_detection = self.loss_func_detection(detection_logits, label)
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
            'loss_detection': loss_detection_test,
            'acc_detection': acc_detection_test,
            'confusion_matrix': cm_detection,
            'classification_report': cr_detection
        }
    
    def train(self, train_loader: DataLoader, test_loader: DataLoader, num_epochs: int = 50):
        """Main training loop following official repository."""
        logger.info("Starting COOLANT training...")
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Evaluate
            test_metrics = self.evaluate(test_loader)
            
            # Print results (following official format)
            logger.info(f'--- TASK1 CLIP ---')
            logger.info(f'[Epoch: {epoch}], losses: {train_metrics["loss_clip"]:.4f}')
            
            logger.info(f'--- TASK2 Detection ---')
            if test_metrics['acc_detection'] > self.best_acc:
                self.best_acc = test_metrics['acc_detection']
                logger.info(f'New best accuracy! Classification Report:\n{test_metrics["classification_report"]}')
                
                # Save best models
                save_dir = Path(self.config.get('save_dir', './checkpoints'))
                save_dir.mkdir(exist_ok=True)
                
                torch.save(self.model.clip_module.state_dict(), save_dir / "best_clip_module.pth")
                torch.save(self.model.detection_module.state_dict(), save_dir / "best_detection_model.pth")
                logger.info(f"Saved best models to {save_dir}")
            
            logger.info(
                f"EPOCH = {epoch + 1}\n"
                f"acc_detection_train = {train_metrics['acc_detection']:.3f}\n"
                f"acc_detection_test = {test_metrics['acc_detection']:.3f}\n"
                f"best_acc = {self.best_acc:.3f}\n"
                f"loss_detection_train = {train_metrics['loss_detection']:.3f}\n"
                f"loss_detection_test = {test_metrics['loss_detection']:.3f}\n"
            )
            
            logger.info(f'--- TASK2 Detection Confusion Matrix ---\n{test_metrics["confusion_matrix"]}\n')
        
        logger.info("Training completed!")


def main():
    """Main training function."""
    # Configuration
    config = {
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'batch_size': 64,
        'lr': 1e-3,
        'l2': 0,
        'num_epochs': 50,
        'seed': 825,
        'save_dir': './checkpoints',
        
        # Model configuration
        'shared_dim': 128,
        'sim_dim': 64,
        'clip_embed_dim': 64,
        'feature_dim': 96,  # 64 + 16 + 16
        'h_dim': 64,
        
        # Data configuration
        'json_path': 'src/data/json/news_data.json',
        'image_base_dir': 'src/data/jpg',
        'preprocessed_dir': 'preprocessed_coolant',
        'label_mapping': {'thanh_nien': 0, 'dan_tri': 1, 'vnexpress': 2, 'tuoitre': 3}
    }
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        json_path=config['json_path'],
        image_base_dir=config['image_base_dir'],
        batch_size=config['batch_size'],
        label_mapping=config['label_mapping'],
        preprocessed_dir=config['preprocessed_dir']
    )
    
    # Initialize trainer
    trainer = COOLANTTrainer(config)
    
    # Start training
    trainer.train(train_loader, test_loader, config['num_epochs'])


if __name__ == "__main__":
    main()
