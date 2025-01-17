import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import io
import base64
from flask import Flask, request, jsonify
from torchvision.models import mobilenet_v3_large

# Flask 앱 생성
app = Flask(__name__)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 전처리 정의
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Pretrained 모델 정의
model = mobilenet_v3_large(pretrained=True)

# 출력 레이어 수정
model.classifier = nn.Sequential(
    nn.Flatten(),  # 텐서를 1차원으로 변환
    nn.Linear(model.classifier[0].in_features, 1)  # 이진 분류를 위한 Fully Connected Layer
)

# 저장된 모델 로드
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()  # 평가 모드 설정

# 모델 예측 함수
def predict_image(model, image):
    """이미지를 모델로 예측"""
    transformed_image = data_transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(transformed_image).squeeze().item()
        prob = torch.sigmoid(torch.tensor(output)).item()  # Sigmoid로 확률 계산
        prediction = "Fire" if prob > 0.5 else "No Fire"
    
    return prediction, prob

@app.route('/predict', methods=['POST'])
def predict():
    """base64 문자열로 이미지 받아 예측"""
    data = request.get_json()

    # base64 문자열에서 이미지 디코딩
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    try:
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        return jsonify({'error': f'Failed to decode image: {str(e)}'}), 400

    # 모델로 예측 수행
    prediction, prob = predict_image(model, image)

    # 결과 반환
    return jsonify({
        'prediction': prediction,
        'probability': round(prob, 4)  # 소수점 4자리까지 반올림
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
