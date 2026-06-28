# Architecture Decisions

This document records the reasoning behind architectural decisions.

The purpose is to prevent future redesigns that remove intentional engineering choices.

---

# Decision 1

Use a Modular Architecture

Decision

Separate every major responsibility into its own module.

Reason

Small modules are easier to test, debug, maintain and extend.

Implemented

AIService

AIEngine

GPUManager

BatchManager

CheckpointManager

ModelLoader

Registry

Upscaler

GPUMonitor

Status

Permanent

---

# Decision 2

GPUManager only manages GPU resources

Decision

GPUManager never processes images.

Reason

Single Responsibility Principle.

GPUManager should only

• Load models

• Unload models

• Reload models

• Clean CUDA memory

• Report GPU statistics

Image enhancement belongs elsewhere.

Status

Permanent

---

# Decision 3

AIEngine owns image enhancement

Decision

AIEngine performs image enhancement.

Reason

Keeps AI processing separate from GPU lifecycle.

Responsibilities

Read image

Call Upscaler

Save output

Nothing more.

Status

Permanent

---

# Decision 4

Upscaler wraps RealESRGAN

Decision

RealESRGAN is hidden behind an Upscaler abstraction.

Reason

Future models can replace the implementation without changing AIEngine.

Future compatibility

GFPGAN

CodeFormer

RIFE

Custom AI models

Status

Permanent

---

# Decision 5

Model loading is centralized

Decision

Only ModelLoader creates AI models.

Reason

Avoid duplicated loading logic.

Changing models should require modifying only one location.

Status

Permanent

---

# Decision 6

Registry manages model names

Decision

Model names are never hardcoded throughout the project.

Reason

One location maps

Model Name

↓

Weights

Future models only require registry updates.

Status

Permanent

---

# Decision 7

Batch processing

Decision

Frames are processed in batches.

Reason

Allows

GPU cleanup

Checkpoint updates

Future parallel processing

Worker process isolation

Status

Permanent

---

# Decision 8

Resume uses the output folder

Decision

Existing enhanced frames are treated as completed work.

Reason

The filesystem represents actual completed work.

Checkpoint may become outdated or corrupted.

Output folder is the primary source of truth.

Checkpoint is secondary.

Status

Permanent

---

# Decision 9

Checkpoint stores metadata

Decision

Checkpoint stores

Movie metadata

AI settings

Progress

Errors

Statistics

Reason

Checkpoint should not be the only source of resume information.

Status

Permanent

---

# Decision 10

GPU diagnostics

Decision

Monitor

Temperature

VRAM

GPU utilization

Power draw

Reason

GPU issues should be diagnosed using measurements rather than assumptions.

Status

Permanent

---

# Decision 11

Batch-level GPU reload

Decision

Reload the AI model after each completed batch.

Reason

Reduces long-running CUDA instability.

Improves robustness.

Status

Experimental

Will be evaluated during stability testing.

---

# Decision 12

Development Philosophy

The project prioritizes

Maintainability

Modularity

Reliability

Recoverability

Readability

over short-term optimization.

Architecture should support future integration of

GFPGAN

CodeFormer

RIFE

Video reconstruction

without major redesign.

Status

Permanent