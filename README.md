# Dynamo OpenShift Inference Lab

OpenShift deployment patterns and validated runbooks for NVIDIA Dynamo
inference workloads.

## Included deployment

- `dynamo-multinode-disagg.yaml` — Dynamo 1.4.2 multi-node, disaggregated
  vLLM deployment using Grove and KAI Scheduler.
- `dynamo-multinode-disaggregated-inference-kai-grove-deployment.md` —
  deployment, validation, and OpenAI-compatible chat instructions.

The current deployment is a development validation on ai-dev02 using
Qwen/Qwen3-0.6B, four A10G GPUs, TCP NCCL, KAI Scheduler, and Grove.
See the runbook before applying the manifest: it includes the required
OpenShift namespace UID-range adjustments and control-plane prerequisites.
