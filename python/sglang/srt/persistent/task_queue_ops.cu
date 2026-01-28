#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INT32(x) TORCH_CHECK(x.scalar_type() == torch::kInt32, #x " must be int32")

namespace {

__global__ void enqueue_kernel(const int* tasks,
                               int num_tasks,
                               int num_fields,
                               int* queue,
                               int capacity,
                               int* tail,
                               int* count,
                               int* status) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_tasks) {
    return;
  }
  int slot = atomicAdd(tail, 1);
  if (slot >= capacity) {
    slot = slot % capacity;
  }
  atomicAdd(count, 1);
  int task_offset = idx * num_fields;
  int queue_offset = slot * num_fields;
  for (int f = 0; f < num_fields; ++f) {
    queue[queue_offset + f] = tasks[task_offset + f];
  }
  status[idx] = 1;
}

__global__ void dequeue_kernel(int* queue,
                               int num_fields,
                               int capacity,
                               int* head,
                               int* count,
                               int* out,
                               int* status) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int old_count = atomicAdd(count, -1);
  if (old_count <= 0) {
    atomicAdd(count, 1);
    status[0] = 0;
    return;
  }
  int slot = atomicAdd(head, 1);
  if (slot >= capacity) {
    slot = slot % capacity;
  }
  int queue_offset = slot * num_fields;
  for (int f = 0; f < num_fields; ++f) {
    out[f] = queue[queue_offset + f];
  }
  status[0] = 1;
}

__global__ void drain_kernel(int* queue,
                             int num_fields,
                             int capacity,
                             int* head,
                             int* count,
                             int* out,
                             int max_tasks,
                             int* out_count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int popped = 0;
  while (popped < max_tasks) {
    int old_count = atomicAdd(count, -1);
    if (old_count <= 0) {
      atomicAdd(count, 1);
      break;
    }
    int slot = atomicAdd(head, 1);
    if (slot >= capacity) {
      slot = slot % capacity;
    }
    int queue_offset = slot * num_fields;
    int out_offset = popped * num_fields;
    for (int f = 0; f < num_fields; ++f) {
      out[out_offset + f] = queue[queue_offset + f];
    }
    popped += 1;
  }
  out_count[0] = popped;
}

__device__ int select_bucket_id(int value, const int* bucket_sizes, int num_buckets) {
  if (num_buckets <= 0) {
    return -1;
  }
  int idx = 0;
  while (idx < num_buckets && bucket_sizes[idx] < value) {
    idx += 1;
  }
  if (idx >= num_buckets) {
    idx = num_buckets - 1;
  }
  return bucket_sizes[idx];
}

__global__ void drain_with_bucket_kernel(int* queue,
                                         int num_fields,
                                         int capacity,
                                         int* head,
                                         int* count,
                                         int* out,
                                         int max_tasks,
                                         int* out_count,
                                         const int* bucket_sizes,
                                         int num_buckets,
                                         int bucket_field,
                                         int select_field) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int popped = 0;
  while (popped < max_tasks) {
    int old_count = atomicAdd(count, -1);
    if (old_count <= 0) {
      atomicAdd(count, 1);
      break;
    }
    int slot = atomicAdd(head, 1);
    if (slot >= capacity) {
      slot = slot % capacity;
    }
    int queue_offset = slot * num_fields;
    int out_offset = popped * num_fields;
    for (int f = 0; f < num_fields; ++f) {
      out[out_offset + f] = queue[queue_offset + f];
    }
    if (bucket_sizes != nullptr && num_buckets > 0) {
      int value = out[out_offset + select_field];
      out[out_offset + bucket_field] = select_bucket_id(
          value, bucket_sizes, num_buckets);
    }
    popped += 1;
  }
  out_count[0] = popped;
}

}  // namespace

void enqueue(torch::Tensor tasks,
             torch::Tensor queue,
             torch::Tensor tail,
             torch::Tensor count,
             int64_t capacity,
             torch::Tensor status) {
  CHECK_CUDA(tasks);
  CHECK_CUDA(queue);
  CHECK_CUDA(tail);
  CHECK_CUDA(count);
  CHECK_CUDA(status);
  CHECK_CONTIGUOUS(tasks);
  CHECK_CONTIGUOUS(queue);
  CHECK_CONTIGUOUS(tail);
  CHECK_CONTIGUOUS(count);
  CHECK_CONTIGUOUS(status);
  CHECK_INT32(tasks);
  CHECK_INT32(queue);
  CHECK_INT32(tail);
  CHECK_INT32(count);
  CHECK_INT32(status);

  int num_tasks = tasks.size(0);
  if (num_tasks == 0) {
    return;
  }
  int num_fields = tasks.size(1);

  const int threads = 256;
  const int blocks = (num_tasks + threads - 1) / threads;
  auto stream = at::cuda::getDefaultCUDAStream();
  enqueue_kernel<<<blocks, threads, 0, stream.stream()>>>(
      tasks.data_ptr<int>(),
      num_tasks,
      num_fields,
      queue.data_ptr<int>(),
      static_cast<int>(capacity),
      tail.data_ptr<int>(),
      count.data_ptr<int>(),
      status.data_ptr<int>());
}

void dequeue(torch::Tensor queue,
             torch::Tensor head,
             torch::Tensor count,
             int64_t capacity,
             torch::Tensor out,
             torch::Tensor status) {
  CHECK_CUDA(queue);
  CHECK_CUDA(head);
  CHECK_CUDA(count);
  CHECK_CUDA(out);
  CHECK_CUDA(status);
  CHECK_CONTIGUOUS(queue);
  CHECK_CONTIGUOUS(head);
  CHECK_CONTIGUOUS(count);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(status);
  CHECK_INT32(queue);
  CHECK_INT32(head);
  CHECK_INT32(count);
  CHECK_INT32(out);
  CHECK_INT32(status);

  int num_fields = queue.size(1);
  auto stream = at::cuda::getDefaultCUDAStream();
  dequeue_kernel<<<1, 1, 0, stream.stream()>>>(
      queue.data_ptr<int>(),
      num_fields,
      static_cast<int>(capacity),
      head.data_ptr<int>(),
      count.data_ptr<int>(),
      out.data_ptr<int>(),
      status.data_ptr<int>());
}

void drain(torch::Tensor queue,
           torch::Tensor head,
           torch::Tensor count,
           int64_t capacity,
           torch::Tensor out,
           torch::Tensor out_count) {
  CHECK_CUDA(queue);
  CHECK_CUDA(head);
  CHECK_CUDA(count);
  CHECK_CUDA(out);
  CHECK_CUDA(out_count);
  CHECK_CONTIGUOUS(queue);
  CHECK_CONTIGUOUS(head);
  CHECK_CONTIGUOUS(count);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(out_count);
  CHECK_INT32(queue);
  CHECK_INT32(head);
  CHECK_INT32(count);
  CHECK_INT32(out);
  CHECK_INT32(out_count);

  int num_fields = queue.size(1);
  int max_tasks = out.size(0);
  auto stream = at::cuda::getDefaultCUDAStream();
  drain_kernel<<<1, 1, 0, stream.stream()>>>(
      queue.data_ptr<int>(),
      num_fields,
      static_cast<int>(capacity),
      head.data_ptr<int>(),
      count.data_ptr<int>(),
      out.data_ptr<int>(),
      max_tasks,
      out_count.data_ptr<int>());
}

void drain_with_bucket(torch::Tensor queue,
                       torch::Tensor head,
                       torch::Tensor count,
                       int64_t capacity,
                       torch::Tensor out,
                       torch::Tensor out_count,
                       torch::Tensor bucket_sizes,
                       int64_t bucket_field,
                       int64_t select_field) {
  CHECK_CUDA(queue);
  CHECK_CUDA(head);
  CHECK_CUDA(count);
  CHECK_CUDA(out);
  CHECK_CUDA(out_count);
  CHECK_CONTIGUOUS(queue);
  CHECK_CONTIGUOUS(head);
  CHECK_CONTIGUOUS(count);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(out_count);
  CHECK_INT32(queue);
  CHECK_INT32(head);
  CHECK_INT32(count);
  CHECK_INT32(out);
  CHECK_INT32(out_count);

  const int* bucket_ptr = nullptr;
  int num_buckets = 0;
  if (bucket_sizes.defined() && bucket_sizes.numel() > 0) {
    CHECK_CUDA(bucket_sizes);
    CHECK_CONTIGUOUS(bucket_sizes);
    CHECK_INT32(bucket_sizes);
    bucket_ptr = bucket_sizes.data_ptr<int>();
    num_buckets = bucket_sizes.numel();
  }

  int num_fields = queue.size(1);
  int max_tasks = out.size(0);
  auto stream = at::cuda::getDefaultCUDAStream();
  drain_with_bucket_kernel<<<1, 1, 0, stream.stream()>>>(
      queue.data_ptr<int>(),
      num_fields,
      static_cast<int>(capacity),
      head.data_ptr<int>(),
      count.data_ptr<int>(),
      out.data_ptr<int>(),
      max_tasks,
      out_count.data_ptr<int>(),
      bucket_ptr,
      num_buckets,
      static_cast<int>(bucket_field),
      static_cast<int>(select_field));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("enqueue", &enqueue, "Persistent task enqueue (CUDA)");
  m.def("dequeue", &dequeue, "Persistent task dequeue (CUDA)");
  m.def("drain", &drain, "Persistent task drain (CUDA)");
  m.def("drain_with_bucket", &drain_with_bucket,
        "Persistent task drain with bucket selection (CUDA)");
}
