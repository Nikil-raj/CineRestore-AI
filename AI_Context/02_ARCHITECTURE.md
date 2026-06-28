# CineRestore AI Architecture

## High Level Pipeline

Video

↓

Extract Frames

↓

Batch Manager

↓

AI Service

↓

AI Engine

↓

Upscaler

↓

RealESRGAN

↓

Enhanced Frames

↓

Video Reconstruction

---

# Architecture

AIService

Responsibilities

• Main orchestrator

• Coordinates all AI processing

• Processes batches

• Updates checkpoint

• Handles resume

• Reloads GPU after each batch

---

BatchManager

Responsibilities

• Finds all frames

• Splits work into batches

• Skips already enhanced frames

• Supports automatic resume

---

CheckpointManager

Responsibilities

• Stores project metadata

• Stores AI configuration

• Stores processing progress

• Stores error history

Checkpoint is now secondary.

The output folder is the primary source of truth.

---

GPUManager

Responsibilities

• Load AI model

• Unload AI model

• Reload AI model

• CUDA cleanup

• GPU memory management

GPUManager never enhances images.

---

ModelLoader

Responsibilities

• Loads RealESRGAN models

• Loads weight files

• Creates the upsampler

---

Registry

Responsibilities

• Knows every AI model

• Maps model name → weight file

Example

general

↓

realesr-general-x4v3.pth

---

Upscaler

Responsibilities

• Wraps RealESRGAN

• Provides

upscale(image)

Only.

No GPU management.

---

AIEngine

Responsibilities

• Reads image

• Calls Upscaler

• Saves enhanced image

Engine never loads models.

---

GPUMonitor

Responsibilities

• Reads GPU temperature

• Reads VRAM usage

• Reads GPU utilization

• Reads GPU power

Uses

nvidia-smi

for diagnostics.

---

Current Processing Flow

BatchManager

↓

AIService

↓

AIEngine

↓

Upscaler

↓

GPUManager

↓

RealESRGAN

↓

Enhanced Frame

↓

Checkpoint

↓

Next Frame