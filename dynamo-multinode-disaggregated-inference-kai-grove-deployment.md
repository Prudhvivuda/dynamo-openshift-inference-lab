# Dynamo multi-node disaggregated inference with KAI and Grove

This runbook documents the deployment validated on the `ai-dev02` OpenShift
cluster on 2026-09-02. It deploys NVIDIA Dynamo 1.4.2 with:

- Grove for multi-node worker orchestration
- KAI Scheduler for gang scheduling
- one frontend
- a two-node prefill worker group
- a two-node decode worker group
- four A10G GPUs total, one GPU per worker pod

The workload manifest is
[`dynamo-multinode-disaggregated-inference-kai-grove.yaml`](./dynamo-multinode-disaggregated-inference-kai-grove.yaml).

## Result

The deployment completed successfully and an OpenAI-compatible completion sent
to the frontend returned a valid response from `Qwen/Qwen3-0.6B`.

The final resources are:

| Resource | Value |
| --- | --- |
| Namespace | `dynamo-multinode-pvuda` |
| DynamoGraphDeployment | `vllm-disagg-multinode` |
| Model | `Qwen/Qwen3-0.6B` |
| Prefill | 2 physical nodes, TP=2 |
| Decode | 2 physical nodes, TP=2 |
| GPU type | NVIDIA A10G |
| GPU count | 4 |

## Prerequisites

1. Authenticate to the intended cluster and verify the current context.

   ```bash
   oc login <api-server> --token=<token>
   oc whoami --show-context
   ```

2. Confirm sufficient GPU capacity. This deployment requires four available
   GPUs plus a CPU-capable node for the frontend.

   ```bash
   oc get nodes -l nvidia.com/gpu.present=true

   oc get pods -A -o json | jq -r '
     .items[] |
     select(any(.spec.containers[]?;
       ((.resources.requests["nvidia.com/gpu"] //
         .resources.limits["nvidia.com/gpu"] // "0") | tonumber) > 0)) |
     [.metadata.namespace, .metadata.name, (.spec.nodeName // "<pending>")] |
     @tsv'
   ```

3. Check GPU-node taints. On ai-dev02 the free A10G nodes use
   `g5-gpu=true:NoSchedule`, so the manifest includes its toleration.

   ```bash
   oc get nodes -l nvidia.com/gpu.present=true -o json | jq -r '
     .items[] |
     [.metadata.name, (.spec.taints // [] |
       map(.key + "=" + (.value // "") + ":" + .effect) | join(";"))] |
     @tsv'
   ```

## 1. Enable Grove and KAI Scheduler in Dynamo Platform

The existing `dynamo-platform` Helm release was upgraded to enable the bundled
Grove and KAI components.

```bash
helm upgrade dynamo-platform dynamo-platform-1.4.2.tgz \
  --namespace dynamo-system \
  --reuse-values \
  --set global.grove.install=true \
  --set global.kai-scheduler.install=true \
  --wait --timeout 10m
```

If the release archive is not already local, obtain the matching 1.4.2 chart
from the NVIDIA Dynamo Helm repository first. Keep the platform, Grove, and KAI
versions aligned.

### OpenShift CRD installation note

Some KAI CRDs exceed the client-side apply annotation limit on OpenShift. Apply
the chart CRDs server-side before retrying the Helm upgrade if that occurs:

```bash
oc apply --server-side -f <extracted-dynamo-chart>/crds/
```

Verify the control plane:

```bash
oc get deployment -n dynamo-system
oc get queue.scheduling.run.ai
```

The deployment uses the `dynamo` KAI queue, which is created by the Dynamo
KAI subchart.

### KAI memory settings used on ai-dev02

The chart defaults caused OOM restarts for the KAI binder and scheduler. The
following persistent KAI `Config` patch was applied:

```bash
oc patch config.kai.scheduler/kai-config --type=merge -p '{
  "spec": {
    "binder": {
      "service": {
        "resources": {
          "requests": {"cpu": "250m", "memory": "1Gi"},
          "limits": {"cpu": "700m", "memory": "1Gi"}
        }
      }
    },
    "scheduler": {
      "service": {
        "resources": {
          "requests": {"cpu": "250m", "memory": "2Gi"},
          "limits": {"cpu": "700m", "memory": "2Gi"}
        }
      }
    }
  }
}'
```

Verify the operator preserves these values:

```bash
oc get deployment binder kai-scheduler-default -n dynamo-system -o json |
  jq -r '.items[] |
    [.metadata.name,
     .spec.template.spec.containers[0].resources.requests.memory,
     .spec.template.spec.containers[0].resources.limits.memory] | @tsv'
```

### Grove finalizer RBAC note

The installed Grove operator role lacked finalizer-subresource update
permissions. The following one-time patch was applied:

```bash
oc patch clusterrole grove:system:grove-operator --type=json -p='[
  {
    "op": "add",
    "path": "/rules/-",
    "value": {
      "apiGroups": ["grove.io"],
      "resources": [
        "podcliquesets/finalizers",
        "podcliquescalinggroups/finalizers",
        "podcliques/finalizers"
      ],
      "verbs": ["update"]
    }
  }
]'

oc rollout restart deployment/grove-operator -n dynamo-system
oc rollout status deployment/grove-operator -n dynamo-system --timeout=90s
```

Recheck this permission after any Dynamo/Grove Helm upgrade because the
Helm-managed ClusterRole may be reconciled.

## 2. Create the workload namespace and use its OpenShift UID range

OpenShift allocates an SCC UID range to each namespace. The manifest used the
base UID `1001330000` assigned to `dynamo-multinode-pvuda` as its `fsGroup`.

```bash
oc create namespace dynamo-multinode-pvuda

oc get namespace dynamo-multinode-pvuda \
  -o jsonpath='{.metadata.annotations.openshift\.io/sa\.scc\.uid-range}{"\n"}'
```

For another namespace, update every `fsGroup` value in the manifest to the base
UID returned by that command before deploying. Do not reuse `1001330000` in a
new namespace.

The workload runs under the normal `restricted-v2` SCC; no `anyuid` SCC was
granted. The manifest sets `runAsNonRoot: true`, and includes `USER=dynamo`,
`HOME=/tmp`, `HF_HOME=/tmp/huggingface`, and
`TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor`. These avoid image assumptions
about a fixed passwd entry while OpenShift assigns the runtime UID.

## 3. Deploy the direct DynamoGraphDeployment

DGDR profiling was intentionally not used. Its profiling jobs hardcode UID/GID
values that conflict with OpenShift namespace-assigned UID ranges. This direct
DGD defines the topology explicitly.

```bash
oc apply -f dynamo-multinode-disaggregated-inference-kai-grove.yaml
```

Watch deployment status:

```bash
oc get dynamographdeployment vllm-disagg-multinode \
  -n dynamo-multinode-pvuda -w

oc get pods -n dynamo-multinode-pvuda -o wide
```

Successful status:

```text
state: successful
Ready: True
message: All resources are ready
```

## 4. Test inference

### Verify graph and pod readiness

```bash
oc get dynamographdeployment vllm-disagg-multinode \
  -n dynamo-multinode-pvuda -o json |
  jq '{state: .status.state, conditions: .status.conditions, components: .status.components}'

oc get pods -n dynamo-multinode-pvuda -o wide
```

Expected result: five `1/1 Running` pods: frontend, prefill leader/worker, and
decode leader/worker.

### Run an in-cluster completion smoke test

Find the current frontend pod; its generated suffix changes after a rollout.

```bash
NS=dynamo-multinode-pvuda
FRONTEND_POD=$(oc get pods -n "$NS" -o name | sed -n '/frontend/p' | head -n 1 | cut -d/ -f2)
echo "$FRONTEND_POD"
```

Send one completion through the frontend:

```bash
oc exec -n "$NS" "$FRONTEND_POD" -- python3 -c '
import json, urllib.request
data = json.dumps({
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "Reply exactly with: OK",
  "max_tokens": 4,
  "temperature": 0
}).encode()
request = urllib.request.Request(
  "http://127.0.0.1:8000/v1/completions",
  data=data,
  headers={"Content-Type": "application/json"}
)
print(urllib.request.urlopen(request, timeout=120).read().decode())
'
```

The validated deployment returned an OpenAI-compatible JSON completion.

### Chat with the model from a local terminal

Start a port-forward and leave it running in one terminal:

```bash
oc port-forward -n dynamo-multinode-pvuda \
  service/vllm-disagg-multinode-frontend 8000:8000
```

In a second terminal, send a chat-completions request:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Explain tensor parallelism in one sentence."}
    ],
    "max_tokens": 96,
    "temperature": 0.2
  }' | jq
```

For a simple completion instead of chat:

```bash
curl -sS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "Explain tensor parallelism in one sentence.",
    "max_tokens": 96,
    "temperature": 0.2
  }' | jq
```

## Constraints and decisions

- This is a functional development validation, not a performance benchmark.
- ai-dev02 has no RDMA configuration. The manifest forces NCCL over TCP with
  `NCCL_SOCKET_IFNAME=eth0` and `NCCL_IB_DISABLE=1`.
- The A10G GPUs provide approximately 22 GB each. The default model context and
  CUDA-graph capture exhausted available GPU memory, so validation uses
  `--max-model-len 4096`, `--gpu-memory-utilization 0.75`, and
  `--enforce-eager`.
- These settings reduce throughput but allow reliable multi-node execution on
  the available dev-cluster capacity.
- For production-scale disaggregated serving, enable and validate RDMA, then
  revisit the vLLM memory, context, and graph-capture settings.

## Reusing this runbook for a new task

Use a different namespace and DGD name. Create the namespace first, retrieve
its assigned UID range, update the manifest `metadata.namespace` and all
`securityContext.fsGroup` values, then apply it. Confirm capacity and taints
before scheduling a new four-GPU gang so it does not disrupt this deployment.
