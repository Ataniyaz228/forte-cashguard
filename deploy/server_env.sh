# RunPod server settings (sourced before deploy/serve_loop.sh)
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
export CASHGUARD_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct CASHGUARD_PORT=8000
export CASHGUARD_WORKERS=2 CASHGUARD_VLM_FPS_LONG=0.5
