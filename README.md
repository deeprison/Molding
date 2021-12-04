# Molding

https://www.notion.so/Molding-RL-21a73fcb3dbc43d783edc8ccd2327193

## Environment
Molding 실험을 위한 간단한 환경 (with Python 3.8)

### 필수 패키지
```
pip install opencv-python numpy albumentations
```

### 테스트
```
python env.py
```

### Linux Rendering
- python 실행 파일에서
```
...
env.render(on_terminal=True)
...
```
- 학습 실행 후 터미널에서
```
watch -d -n 0.01 ./render/render.log
```