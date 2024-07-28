from tqdm import tqdm
import cv2 as cv
import torch
def draw_labels(data_loader, labels_predicted, classes, scale, offset):
    for images, labels in tqdm(data_loader):
        images = images.to(labels_predicted.device)
        labels = labels.to(labels_predicted.device)
        total += labels.size(0)
        correct += (labels_predicted == labels).sum().item()
        idx = 0
        img = images[idx].numpy()
        img = img.transpose(1, 2, 0)
        img *= scale
        img += offset
        img = cv.copyMakeBorder(img, 0, 20, 0, 0, cv.BORDER_CONSTANT, value=[255, 255, 255])
        cv.putText(img, classes[labels_predicted[idx]], (img.shape[1] // 2 - 10, img.shape[0] - 5),
                        cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv.LINE_AA)
        examples += [torch.tensor(img.transpose(2, 0, 1))]
        