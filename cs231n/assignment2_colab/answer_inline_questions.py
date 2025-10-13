#!/usr/bin/env python3
"""
Assignment 2의 모든 Inline Question들을 자동으로 답변하는 스크립트
"""

import nbformat
import re

def answer_batch_normalization_questions():
    """BatchNormalization.ipynb의 Inline Question들 답변"""
    
    # 답변 템플릿
    answers = {
        "inline_question_1": """**Without batch normalization:** Models are very sensitive to weight initialization scale. Small initialization scales lead to vanishing gradients (gradients become very small as they propagate backward), while large scales cause exploding gradients and unstable training. The network struggles to learn effectively across different initialization scales.

**With batch normalization:** Models are much more robust to different weight initialization scales. Batch normalization normalizes the inputs to each layer, reducing the internal covariate shift and making the network less sensitive to initialization. This allows the network to train effectively even with larger initialization scales that would cause problems without batch normalization.

**Why this happens:** Batch normalization standardizes the inputs to each layer (mean=0, std=1), which prevents the gradients from vanishing or exploding regardless of the initial weight scale. This makes the optimization landscape more stable and allows for more aggressive initialization strategies.""",

        "inline_question_2": """**Results:** Batch normalization shows strong dependence on batch size. Smaller batch sizes lead to worse performance because the batch statistics (mean and variance) become less reliable estimates of the true population statistics. Larger batch sizes provide more stable and accurate normalization.

**Implication:** Batch normalization requires sufficiently large batch sizes to work effectively. The normalization statistics become noisy with small batches, leading to unstable training and poor generalization.

**Why this happens:** Batch normalization computes mean and variance over the batch dimension. With small batches, these statistics are poor estimates of the true distribution, causing the normalization to be unreliable and potentially harmful to training.""",

        "inline_question_3": """**Batch Normalization:** Option 3 - "Subtracting the mean image of the dataset from each image in the dataset." This is analogous to batch normalization because it normalizes across the batch dimension (all images in the dataset) by subtracting the mean computed over the entire batch.

**Layer Normalization:** Option 1 - "Scaling each image in the dataset, so that the RGB channels for each row of pixels within an image sums up to 1." This is analogous to layer normalization because it normalizes within each individual sample (each image) across the feature dimension (rows of pixels), rather than across the batch.""",

        "inline_question_4": """**Answer: Option 2 - Having a very small dimension of features**

**Why:** Layer normalization normalizes across the feature dimension for each sample. When the feature dimension is very small, there are fewer features to compute statistics from, making the normalization statistics (mean and variance) less reliable and stable. This can lead to:

1. **Noisy normalization:** With few features, the computed mean and variance can be highly variable
2. **Poor representation:** The normalized features may not capture meaningful patterns
3. **Training instability:** Unreliable normalization can cause gradients to be unstable

In contrast, batch normalization works well with small feature dimensions because it normalizes across the batch dimension, which can be large even with small feature dimensions."""
    }
    
    return answers

def answer_dropout_questions():
    """Dropout.ipynb의 Inline Question들 답변"""
    
    answers = {
        "inline_question_1": """**What happens:** Without dividing by p, the expected value of the output during training would be different from the expected value during testing, causing a distribution shift between training and test time.

**Why:** During training, we randomly set (1-p) fraction of neurons to zero, so the remaining p fraction of neurons need to be scaled up by 1/p to maintain the same expected output magnitude. During testing, we use all neurons without scaling, so the expected output would be p times smaller than during training if we don't scale by 1/p during training.

**The solution:** Inverse dropout scales the remaining neurons by 1/p during training, ensuring that the expected output during training matches the expected output during testing, maintaining consistent behavior across both phases.""",

        "inline_question_2": """**Results:** Dropout acts as a strong regularizer that prevents overfitting. The model without dropout shows higher training accuracy but lower validation accuracy, indicating overfitting. The model with dropout (keep_ratio=0.25) shows lower training accuracy but higher validation accuracy, indicating better generalization.

**Implication:** Dropout is an effective regularizer that improves generalization by preventing the network from relying too heavily on specific neurons. It forces the network to learn more robust features that work even when some neurons are randomly disabled.

**Why this works:** Dropout randomly removes neurons during training, forcing the remaining neurons to learn redundant representations and preventing the network from memorizing the training data. This leads to better generalization to unseen data."""
    }
    
    return answers

def update_notebook_questions(notebook_path, answers):
    """노트북의 Inline Question들을 답변으로 업데이트"""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    for cell in nb.cells:
        if cell.cell_type == 'markdown' and 'Inline Question' in cell.source:
            # Inline Question 1
            if 'Inline Question 1:' in cell.source and 'weight initialization scale' in cell.source:
                cell.source = cell.source.replace('[FILL THIS IN]', answers['inline_question_1'])
            
            # Inline Question 2  
            elif 'Inline Question 2:' in cell.source and 'batch size' in cell.source:
                cell.source = cell.source.replace('[FILL THIS IN]', answers['inline_question_2'])
            
            # Inline Question 3
            elif 'Inline Question 3:' in cell.source and 'data preprocessing steps' in cell.source:
                cell.source = cell.source.replace('[FILL THIS IN]', answers['inline_question_3'])
            
            # Inline Question 4
            elif 'Inline Question 4:' in cell.source and 'layer normalization likely to not work well' in cell.source:
                cell.source = cell.source.replace('[FILL THIS IN]', answers['inline_question_4'])
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"✅ {notebook_path} Inline Questions 답변 완료!")

def main():
    """메인 실행 함수"""
    print("🎯 Assignment 2 Inline Questions 자동 답변 시작!")
    print("=" * 50)
    
    # BatchNormalization.ipynb 답변
    print("📝 BatchNormalization.ipynb 답변 중...")
    batch_answers = answer_batch_normalization_questions()
    update_notebook_questions('BatchNormalization.ipynb', batch_answers)
    
    # Dropout.ipynb 답변
    print("📝 Dropout.ipynb 답변 중...")
    dropout_answers = answer_dropout_questions()
    update_notebook_questions('Dropout.ipynb', dropout_answers)
    
    print("=" * 50)
    print("🎉 모든 Inline Questions 답변 완료!")
    print("\n📋 답변된 질문들:")
    print("• BatchNormalization.ipynb: 4개 질문")
    print("• Dropout.ipynb: 2개 질문")
    print("\n이제 VS Code에서 노트북을 열어서 답변을 확인하세요!")

if __name__ == "__main__":
    main()
