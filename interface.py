import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
st.set_page_config(initial_sidebar_state="collapsed")
st.title("Signature Verification")
upload_file_0 = st.file_uploader("Upload Original Signature", type=["jpg", "jpeg", "png"])
upload_file_1 = st.file_uploader("Upload Signature To Be Verified", type=["jpg", "jpeg", "png"])
if upload_file_0:
    image_original = Image.open(upload_file_0).convert("RGB")
if upload_file_1:
    image_checked = Image.open(upload_file_1).convert("RGB")
model_parameters_path = "trained_model_parameters/siamese_resnet_18.pth"
class SiameseResNet(nn.Module):
    def __init__(self):
        super(SiameseResNet, self).__init__()
        base_model = models.resnet18(weights=None)
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = base_model.fc.in_features
        base_model.fc = nn.Identity()
        self.base_model = base_model
        self.embedding = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    def forward_once(self, x):
        x = self.base_model(x)
        x = self.embedding(x)
        return x
    def forward(self, img1, img2):
        out1 = self.forward_once(img1)
        out2 = self.forward_once(img2)
        return out1, out2
model = SiameseResNet()
model.load_state_dict(torch.load(model_parameters_path, map_location="cpu"))
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
def predict_similarity(img1, img2):
    t1 = transform(img1).unsqueeze(0)
    t2 = transform(img2).unsqueeze(0)
    with torch.no_grad():
        out1, out2 = model(t1, t2)
        distance = torch.nn.functional.pairwise_distance(out1, out2)
    return distance.item()
if st.button("Check"):
    if upload_file_0 and upload_file_1:
        distance = predict_similarity(image_original, image_checked)
        st.session_state["similarity_score"] = distance
        st.session_state["match"] = distance < 0.4275
        st.switch_page("./pages/result_page.py")
    else:
        st.error("Please upload both images first.")
