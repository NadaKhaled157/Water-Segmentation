# Water-Segmentation
Segmenting water bodies using multispectral and optical data. This solution is vital for monitoring water resources, flood management, and environmental conservation, where precise segmentation of water bodies can significantly impact decision-making.

## Twelve-Band Visualization
<img width="1405" height="990" alt="image" src="https://github.com/user-attachments/assets/21ea16ad-1845-42ae-9eb1-64c14df5f7f5" />

## From Scratch
Each convolution block had:
- Two 3x3 convolutional layers.
- Batch normalization after each convolution.
- ReLU activation
- 30% Dropout
### Model
1. Encoder: Four convolutional blocks with increasing channels, followed by 2x2 maxpooling to downsample.
2. Bottleneck
3. Decoder: Four upsampling blocks with decreasing channels.
### Evaluation Metrics
- Loss: 0.1739
- IoU: 0.5859
- Precision: 0.8620
- Recall: 0.7903
- F1: 0.8202
## Pretrained
Used ResNet34 backbone pretrained on ImageNet
### Evaluation Metrics
- Loss: 0.1464
- IoU: 0.6164
- Precision: 0.8648
- Recall: 0.8618
- F1: 0.8564
## Deployment
<img width="625" height="335" alt="image" src="https://github.com/user-attachments/assets/5968f02d-602f-4649-a230-e2d6b212b286" />
<img width="800" height="454" alt="image" src="https://github.com/user-attachments/assets/e05b68c1-66a8-4658-a35e-675cf8a66700" />

