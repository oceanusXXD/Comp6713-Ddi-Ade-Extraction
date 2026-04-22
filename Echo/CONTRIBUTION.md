# Contribution Statement

This contribution statement summarises the division of work for the project submission. The project was completed collaboratively, with each member taking primary responsibility for a specific part of the workflow while also participating in group discussion, experiment review, and final integration.

## Suggested Summary

The project was completed across five closely related workstreams:

- dataset preparation and data auditing
- baseline design and conventional machine learning experiments
- parameter-efficient fine-tuning and training configuration
- inference, evaluation, and error analysis
- demo development, report integration, and result presentation

Although each member had a main area of responsibility, important project decisions, including task definition, experiment scope, model selection, and final result interpretation, were discussed collectively.

## Allocation

| Member     | zID      | Main responsibilities                                                                                                                                                                  | Estimated contribution |
| ---------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| Weilin Guo | z5557085 | dataset preparation and auditing; testing base models including Qwen3B-8B and Llama-8B-Instruct base models; helping verify dataset suitability for downstream training and evaluation | 20%                    |
| Yihao Xu   | z5652298 | baseline implementation using traditional machine learning methods; baseline experiment setup; testing Qwen3B-4B base model for comparison with other approaches                       | 20%                    |
| Tao Zhong  | z5643577 | RS-LoRA training pipeline; fine-tuning configuration and execution; post-training inference and testing to verify trained model behaviour                                              | 20%                    |
| Huping Xue | z5541828 | evaluation and analysis, including Precision, Recall, and F1-score calculation; identification and review of misclassified instances; common error type analysis and interpretation    | 20%                    |
| Zepeng Fan | z5575171 | Gradio demo development; inference testing and usability verification; supporting final result presentation and system demonstration                                                   | 20%                    |

## Detailed Contribution Notes

### Weilin Guo (z5557085)

Weilin Guo was mainly responsible for the data-related and initial model testing components of the project. This included preparing and checking the dataset, identifying issues relevant to model input quality, and supporting dataset readiness for subsequent experiments. Weilin also conducted base-model testing with Qwen3B-8B and Llama-8B-Instruct base models, which helped establish an early understanding of model behaviour and provided reference points for later comparisons.

### Yihao Xu (z5652298)

Yihao Xu was mainly responsible for the baseline study. This included designing and implementing conventional machine learning baselines, running baseline experiments, and organising comparison settings so that the stronger neural approaches could be evaluated against simpler reference methods. Yihao also tested the Qwen3B-4B base model, contributing additional baseline evidence for comparing pretrained model performance before fine-tuning.

### Tao Zhong (z5643577)

Tao Zhong was mainly responsible for training-related work, especially RS-LoRA fine-tuning. Tao handled training configuration, execution of fine-tuning experiments, and related tuning work needed to obtain stable model outputs. Tao also conducted inference tests on trained checkpoints to examine whether the fine-tuned model behaved as expected and to support subsequent evaluation.

### Huping Xue (z5541828)

Huping Xue was mainly responsible for evaluation and error analysis. This included computing key classification metrics such as Precision, Recall, and F1-score, reviewing misclassified instances, and analysing common error types across predictions. This work was important for understanding not only overall model performance but also the main limitations and failure patterns of the final system.

### Zepeng Fan (z5575171)

Zepeng Fan was mainly responsible for the interactive demo and additional inference verification. This included building the Gradio demo for presenting the system in a more accessible way, testing inference behaviour from a user-facing perspective, and helping ensure that the final project outputs could be demonstrated clearly and consistently. Zepeng also contributed to recording demo results and supporting the final presentation of findings.

## Collaborative Aspects

- All members participated in discussions about the task definition, label schema, experiment scope, and final methodology.
- Experimental findings were shared and reviewed within the group before final conclusions were written.
- The final report and submission materials were integrated collaboratively, with each member contributing content or verification related to their workstream.
- Final decisions on which results to report and how to interpret them were made jointly based on group discussion.

## Final Note

The percentages above are estimated to reflect an approximately balanced contribution across complementary tasks. While responsibilities were divided by primary workstream, the project outcome depended on close coordination between dataset preparation, baseline comparison, training, evaluation, and system demonstration.
