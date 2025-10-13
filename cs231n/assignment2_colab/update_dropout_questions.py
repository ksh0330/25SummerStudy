#!/usr/bin/env python3
"""
Dropout.ipynb의 Inline Question들을 답변하는 스크립트
"""

import nbformat
import re

def update_dropout_questions():
    """Dropout.ipynb의 Inline Question들을 답변으로 업데이트"""
    
    with open('Dropout.ipynb', 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # 답변 템플릿
    answers = {
        "inline_question_1": """**What happens:** Without dividing by p, the expected value of the output during training would be different from the expected value during testing, causing a distribution shift between training and test time.

**Why:** During training, we randomly set (1-p) fraction of neurons to zero, so the remaining p fraction of neurons need to be scaled up by 1/p to maintain the same expected output magnitude. During testing, we use all neurons without scaling, so the expected output would be p times smaller than during training if we don't scale by 1/p during training.

**The solution:** Inverse dropout scales the remaining neurons by 1/p during training, ensuring that the expected output during training matches the expected output during testing, maintaining consistent behavior across both phases.""",

        "inline_question_2": """**Results:** Dropout acts as a strong regularizer that prevents overfitting. The model without dropout shows higher training accuracy but lower validation accuracy, indicating overfitting. The model with dropout (keep_ratio=0.25) shows lower training accuracy but higher validation accuracy, indicating better generalization.

**Implication:** Dropout is an effective regularizer that improves generalization by preventing the network from relying too heavily on specific neurons. It forces the network to learn more robust features that work even when some neurons are randomly disabled.

**Why this works:** Dropout randomly removes neurons during training, forcing the remaining neurons to learn redundant representations and preventing the network from memorizing the training data. This leads to better generalization to unseen data."""
    }
    
    for cell in nb.cells:
        if cell.cell_type == 'markdown' and 'Inline Question' in cell.source:
            # Inline Question 1
            if 'Inline Question 1:' in cell.source and 'inverse dropout' in cell.source:
                cell.source = cell.source.replace('[FILL THIS IN]', answers['inline_question_1'])
                print("✅ Inline Question 1 답변 완료")
            
            # Inline Question 2  
            elif 'Inline Question 2:' in cell.source and 'validation and training accuracies' in cell.source:
                cell.source = cell.source.replace('[FILL THIS IN]', answers['inline_question_2'])
                print("✅ Inline Question 2 답변 완료")
    
    with open('Dropout.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print("🎉 Dropout.ipynb Inline Questions 답변 완료!")

if __name__ == "__main__":
    update_dropout_questions()
