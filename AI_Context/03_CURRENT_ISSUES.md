# Current Issues

## Primary Issue

Long-running RealESRGAN processing occasionally crashes.

---

Observed Errors

• CUDA illegal memory access

• CUDA illegal instruction

• CUDNN_STATUS_EXECUTION_FAILED

---

Already Implemented

✅ CUDA cleanup

✅ gc.collect()

✅ torch.cuda.empty_cache()

✅ Model reload

✅ Batch processing

✅ Resume support

✅ Output-folder based resume

✅ GPU monitoring

---

Current GPU Observations

Temperature

76–79°C

VRAM

500–900 MB

Power

32–35 W

No evidence of VRAM exhaustion.

No evidence of overheating.

---

Current Hypothesis

Problem likely originates inside

RealESRGAN

PyTorch

cuDNN

or CUDA runtime

rather than application architecture.

---

Next Investigation

• Monitor long-running stability

• Capture full traceback

• Evaluate tile size

• Consider worker-process isolation if crashes continue