# Project Folder Structure

Cinerestore-AI/

AI_Context/
│
├── 01_PROJECT_STATUS.md
├── 02_ARCHITECTURE.md
├── 03_CURRENT_ISSUES.md
├── 04_FOLDER_STRUCTURE.md
├── 05_NEXT_SPRINT.md
├── 06_TEST_RESULTS.md
├── 07_CHANGELOG.md
└── AI_HANDOVER.md

src/
│
├── ai/
│   ├── engine.py
│   ├── gpu_manager.py
│   ├── model_loader.py
│   ├── registry.py
│   └── upscaler.py
│
├── services/
│   ├── ai_service.py
│   ├── batch_manager.py
│   ├── checkpoint_manager.py
│   └── gpu_monitor.py
│
├── ui/
│
└── utils/

models/

RealESRGAN/

weights/

temp/

frames/

enhanced_frames/

service_test/

service_checkpoint.json

output/

Final restored videos
