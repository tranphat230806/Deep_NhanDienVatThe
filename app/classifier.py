import torch
from torchvision.models import resnet50, ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = ResNet50_Weights.DEFAULT

model = resnet50(weights=weights)
model = model.to(device)
model.eval()

preprocess = weights.transforms()
categories = weights.meta["categories"]


def predict(image):

    img = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)

    probs = torch.softmax(output, dim=1)

    top3 = torch.topk(probs, 3)

    results = []

    for score, idx in zip(top3.values[0], top3.indices[0]):

        results.append({
            "label": categories[idx.item()],
            "confidence": round(score.item() * 100, 2)
        })

    return results