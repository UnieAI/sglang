#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

void replay_graph(py::object graph) {
  py::object target = graph;
  if (py::hasattr(graph, "_graph")) {
    target = graph.attr("_graph");
  }
  if (py::hasattr(target, "replay")) {
    target.attr("replay")();
    return;
  }
  if (py::hasattr(graph, "replay")) {
    graph.attr("replay")();
    return;
  }
  throw std::runtime_error("Persistent graph launcher could not find replay().");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("replay_graph", &replay_graph, "Replay a CUDA graph");
}
