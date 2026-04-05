## ADDED Requirements

### Requirement: Multi-task Training Orchestration
The system SHALL execute three distinct training tasks in each epoch: Similarity Learning, CLIP Contrastive Learning, and Detection with Ambiguity Learning.

#### Scenario: Sequential task execution
- **WHEN** the training loop starts for a batch
- **THEN** the system MUST sequentially update the Similarity Module, then the CLIP Module, and finally the Detection Module

### Requirement: Multi-component Loss Calculation
The system SHALL compute and track separate loss components: `loss_similarity` (Cosine Embedding), `loss_clip` (Cross Entropy with soft labels), `loss_detection` (Cross Entropy), and `loss_skl` (KL Divergence for ambiguity).

#### Scenario: Loss backward propagation
- **WHEN** a task-specific loss is calculated
- **THEN** the system MUST perform backpropagation and update only the parameters associated with that task's optimizer

### Requirement: Experiment Tracking with MLflow
The system SHALL log all training hyperparameters, epoch-level metrics (train/val loss and accuracy), and the best model checkpoints to an MLflow experiment.

#### Scenario: Metric logging
- **WHEN** an epoch completes
- **THEN** the system MUST log `train_loss`, `train_acc`, `val_loss`, and `val_acc` to the active MLflow run

### Requirement: Performance Evaluation
The system SHALL evaluate the model on a test set and generate a confusion matrix and a weighted classification report (Precision, Recall, F1-Score).

#### Scenario: Final evaluation
- **WHEN** training is complete
- **THEN** the system MUST output a detailed classification report and save it as a JSON artifact
