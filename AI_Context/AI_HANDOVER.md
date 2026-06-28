# AI Handover

Read these files before suggesting any code.

Order

1.

01_PROJECT_STATUS.md

2.

02_ARCHITECTURE.md

3.

03_CURRENT_ISSUES.md

4.

05_NEXT_SPRINT.md

Rules

Do NOT redesign the architecture.

Keep the existing modular design.

Respect the current responsibilities of

AIService

AIEngine

GPUManager

BatchManager

CheckpointManager

GPUMonitor

Current Goal

Finish a stable RealESRGAN pipeline.

Only after Sprint 1 is complete should work begin on

GFPGAN

CodeFormer

RIFE

Current Known Issue

Occasional CUDA/cuDNN runtime failures during long-running enhancement.

Resume system is already implemented.

Batch reload is already implemented.

GPU diagnostics are already implemented.

When suggesting fixes,

do not suggest features that already exist.
