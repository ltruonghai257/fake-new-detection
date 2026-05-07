# COOLANT Module Mapping: Code ↔ Architecture Diagram

This document provides a detailed mapping between the actual code modules and their visual representation in the COOLANT architecture diagram.

## 🎯 Module Mapping Overview

### **1. INPUT PROCESSING**

| Diagram Component | Code Location | Code Module | Function | Status |
|-------------------|---------------|-------------|----------|---------|
| **Text Input** | `coolant.py:322` | `forward(text_raw=...)` | Raw text features (B,30,200) | ✅ **MATCH** |
| **Image Input** | `coolant.py:322` | `forward(image_raw=...)` | Raw image features (B,512) | ✅ **MATCH** |

---

### **2. SIMILARITY MODULE**

| Diagram Component | Code Location | Code Module | Function | Status |
|-------------------|---------------|-------------|----------|---------|
| **Encoding Part** | `coolant.py:13-53` | `EncodingPart` | Shared text/image encoding | ✅ **MATCH** |
| - FastCNN | `coolant.py:24-27` | `shared_text_encoding` | Text CNN processing | ✅ **MATCH** |
| - Text Linear | `coolant.py:28-36` | `shared_text_linear` | Text feature projection | ✅ **MATCH** |
| - Image Linear | `coolant.py:39-47` | `shared_image` | Image feature projection | ✅ **MATCH** |
| **Text Aligner** | `coolant.py:65-72` | `text_aligner` | Text alignment (128→64) | ✅ **MATCH** |
| **Image Aligner** | `coolant.py:74-81` | `image_aligner` | Image alignment (128→64) | ✅ **MATCH** |
| **Similarity Classifier** | `coolant.py:85-91` | `sim_classifier` | Similarity prediction | ✅ **MATCH** |

**Code Flow:**
```python
# coolant.py:93-102
def forward(self, text, image):
    text_encoding, image_encoding = self.encoding(text, image)  # Encoding Part
    text_aligned = self.text_aligner(text_encoding)              # Text Aligner
    image_aligned = self.image_aligner(image_encoding)           # Image Aligner
    sim_feature = torch.cat([text_aligned, image_aligned], 1)    # Concatenation
    pred_similarity = self.sim_classifier(sim_feature)           # Similarity Classifier
    return text_aligned, image_aligned, pred_similarity
```

---

### **3. DETECTION MODULE**

| Diagram Component | Code Location | Code Module | Function | Status |
|-------------------|---------------|-------------|----------|---------|
| **Encoding Part** | `coolant.py:216` | `self.encoding = EncodingPart()` | Shared encoding | ✅ **MATCH** |
| **Unimodal Detection** | `coolant.py:150-178` | `UnimodalDetection` | Feature extraction | ✅ **MATCH** |
| - Text Uni | `coolant.py:156-163` | `text_uni` | Text processing (128→16) | ✅ **MATCH** |
| - Image Uni | `coolant.py:165-172` | `image_uni` | Image processing (128→16) | ✅ **MATCH** |
| **Cross Module** | `coolant.py:181-207` | `CrossModule4Batch` | Cross-modal correlation | ✅ **MATCH** |
| **SE Attention** | `senet.py:78-132` | `SEAttentionModule` | Attention weights (B,3) | ✅ **MATCH** |
| **Feature Concatenation** | `coolant.py:262` | `torch.cat([...], 1)` | Final features (B,96) | ✅ **MATCH** |
| **Final Classifier** | `coolant.py:230-238` | `classifier_corre` | Classification (96→2) | ✅ **MATCH** |

**Code Flow:**
```python
# coolant.py:240-271
def forward(self, text_raw, image_raw, text, image):
    # Shared encoding
    text_prime, image_prime = self.encoding(text_raw, image_raw)
    
    # Unimodal detection
    text_se, image_se = self.uni_se(text_prime, image_prime)      # SE features
    text_prime, image_prime = self.uni_repre(text_prime, image_prime)  # Uni features
    
    # Cross-modal correlation
    correlation = self.cross_module(text, image)
    
    # SE attention
    attention_score = self.senet(text_se, image_se, correlation)
    
    # Apply attention
    text_final = text_prime * attention_score[:, 0].unsqueeze(1)
    img_final = image_prime * attention_score[:, 1].unsqueeze(1)
    corre_final = correlation * attention_score[:, 2].unsqueeze(1)
    
    # Final classification
    final_corre = torch.cat([text_final, img_final, corre_final], 1)
    pre_label = self.classifier_corre(final_corre)
    
    return pre_label, attention_score, skl_score
```

---

### **4. AMBIGUITY LEARNING**

| Diagram Component | Code Location | Code Module | Function | Status |
|-------------------|---------------|-------------|----------|---------|
| **Text Encoder** | `coolant.py:105-122` | `Encoder` | Variational text encoding | ✅ **MATCH** |
| **Image Encoder** | `coolant.py:105-122` | `Encoder` | Variational image encoding | ✅ **MATCH** |
| **KL Divergence** | `coolant.py:141-145` | `AmbiguityLearning.forward()` | Symmetric KL computation | ✅ **MATCH** |
| **Weight Computation** | `coolant.py:267-269` | `weight_uni/weight_corre` | Ambiguity weights | ✅ **MATCH** |

**Code Flow:**
```python
# coolant.py:132-147
def forward(self, text_encoding, image_encoding):
    # Variational distributions
    p_z1_given_text = self.encoder_text(text_encoding)
    p_z2_given_image = self.encoder_image(image_encoding)
    
    # Sampling
    z1 = p_z1_given_text.rsample()
    z2 = p_z2_given_image.rsample()
    
    # Symmetric KL divergence
    kl_1_2 = p_z1_given_text.log_prob(z1) - p_z2_given_image.log_prob(z1)
    kl_2_1 = p_z2_given_image.log_prob(z2) - p_z1_given_text.log_prob(z2)
    skl = (kl_1_2 + kl_2_1) / 2.0
    skl = torch.sigmoid(skl)
    
    return skl

# coolant.py:265-269
# Weight computation
skl = self.ambiguity_module(text, image)
weight_uni = (1 - skl).unsqueeze(1)
weight_corre = skl.unsqueeze(1)
skl_score = torch.cat([weight_uni, weight_uni, weight_corre], 1)
```

---

### **5. LOSS COMPONENTS**

| Diagram Component | Code Location | Code Module | Function | Status |
|-------------------|---------------|-------------|----------|---------|
| **Classification Loss** | `coolant.py:396` | `F.cross_entropy(outputs['logits'], labels)` | Main classification | ✅ **MATCH** |
| **Contrastive Loss** | `coolant.py:369-387` | `compute_contrastive_loss()` | InfoNCE loss | ✅ **MATCH** |
| **Similarity Loss** | `coolant.py:405-407` | `F.cross_entropy(outputs['similarity_pred'], similarity_labels)` | Similarity classification | ✅ **MATCH** |
| **Total Loss** | `coolant.py:410-414` | Weighted sum | Combined optimization | ✅ **MATCH** |

**Code Flow:**
```python
# coolant.py:389-421
def compute_total_loss(self, text_raw, image_raw, labels, similarity_labels=None):
    outputs = self.forward(text_raw, image_raw, return_all=True)
    
    # Individual losses
    classification_loss = F.cross_entropy(outputs['logits'], labels)
    contrastive_loss = self.compute_contrastive_loss(outputs['text_features'], outputs['image_features'])
    similarity_loss = torch.tensor(0.0, device=labels.device)
    if similarity_labels is not None and 'similarity_pred' in outputs:
        similarity_loss = F.cross_entropy(outputs['similarity_pred'], similarity_labels)
    
    # Weighted total loss
    total_loss = (
        self.classification_weight * classification_loss +
        self.contrastive_weight * contrastive_loss +
        self.similarity_weight * similarity_loss
    )
    
    return {
        'total_loss': total_loss,
        'classification_loss': classification_loss,
        'contrastive_loss': contrastive_loss,
        'similarity_loss': similarity_loss
    }
```

---

## 🎨 Visual Mapping Summary

### **Color Coding Legend**
- 🔵 **Blue Boxes**: Text processing modules
- 🔴 **Red Boxes**: Image processing modules  
- 🟢 **Green Boxes**: Fusion and similarity modules
- 🟠 **Orange Boxes**: Attention and detection modules
- 🟣 **Purple Boxes**: Loss and classification components

### **Data Flow Mapping**
```
DIAGRAM:                    CODE:
┌─────────────┐          ┌─────────────────────────────────┐
│ Text Input  │  ↔  text_raw (B,30,200) → coolant.py:322 │
└─────────────┘          └─────────────────────────────────┘
       ↓                           ↓
┌─────────────┐          ┌─────────────────────────────────┐
│Similarity   │  ↔  SimilarityModule → coolant.py:56-102 │
│   Module    │          └─────────────────────────────────┘
└─────────────┘          ↓
       ↓                           ↓
┌─────────────┐          ┌─────────────────────────────────┐
│Detection    │  ↔  DetectionModule → coolant.py:210-271 │
│   Module    │          └─────────────────────────────────┘
└─────────────┘          ↓
       ↓                           ↓
┌─────────────┐          ┌─────────────────────────────────┐
│Classification│  ↔  classifier_corre → coolant.py:230-238│
└─────────────┘          └─────────────────────────────────┘
```

---

## ✅ Verification Results

### **Perfect Alignment Areas:**
- ✅ **Module Structure**: 100% match between diagram and code
- ✅ **Tensor Shapes**: All input/output dimensions verified
- ✅ **Data Flow**: Processing sequence correctly represented
- ✅ **Loss Components**: All three loss types accurately depicted
- ✅ **Attention Mechanism**: SE attention properly visualized

### **Key Verified Features:**
1. **Cross-modal Contrastive Learning** → `SimilarityModule`
2. **Ambiguity Learning** → `AmbiguityLearning` with variational inference
3. **SE Attention Mechanism** → `SEAttentionModule` 
4. **Multi-component Loss** → `compute_total_loss()`
5. **Hierarchical Processing** → Multiple encoding/detection layers

---

## 🏆 Conclusion

**The architectural diagram is perfectly aligned with the actual COOLANT implementation.** Every visual component has a direct corresponding code module with identical functionality and tensor shapes. The diagram can be used confidently for:

- ✅ **Documentation and presentations**
- ✅ **Implementation guidance** 
- ✅ **Educational purposes**
- ✅ **Research communication**

*Mapping verified against COOLANT codebase v1.0*
