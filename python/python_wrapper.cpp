#include <CEOs.h>
#include <streamCEOs.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace python {

namespace py = pybind11;

PYBIND11_MODULE(CEOs, m) { // Must be the same name with class CEOs
    py::class_<CEOs>(m, "CEOs")
        .def(py::init<const int&, const int&>(),
           py::arg("n_points"),
           py::arg("n_features"),
         "Constructor for CEOs.\n"
         "n_points: number of points.\n"
         "n_features: number of dimensions."
         )
        //
        .def("setIndexParam", &CEOs::set_CEOsParam,
            py::arg("n_proj"), py::arg("n_repeats") = 1, py::arg("top_m") = 10,
            py::arg("n_threads") = -1, py::arg("random_seed") = -1,
         "n_proj: number pf projections.\n"
         "n_repeats: number of call random projections, so in total we have n_proj * n_repeats number of projections.\n"
         "top-m: number of top-m points closest/furthest to the random vector.\n"
         "n_threads: number of threads."
        )
//        .def_readwrite("n_threads", &CEOs::n_threads)
        .def_readwrite("n_probed_vectors", &CEOs::n_probed_vectors, "Number of probed vectors among n_repeats * n_proj random vectors.")
        .def_readwrite("n_probed_points", &CEOs::n_probed_points, "Number of probed points for each random vector. This must be smaller than top-m")
        .def_readwrite("n_cand", &CEOs::n_cand, "Number of candidates to be selected for computing dot product each query.")
        .def_readwrite("centering", &CEOs::centering, "Set centering, default is False.")
        .def("set_threads", &CEOs::set_threads, py::arg("n_threads"))
        .def("clear", &CEOs::clear)
        //
        .def("build", &CEOs::build_CEOs, py::arg("dataset"))
        .def("search", &CEOs::search_CEOs,
             py::arg("queries"), py::arg("n_neighbors"), py::arg("verbose") = false)
        //
        .def("build_coCEOs_Est", &CEOs::build_coCEOs_Est, py::arg("dataset"))
        .def("search_coCEOs_Est", &CEOs::search_coCEOs_Est,
            py::arg("queries"), py::arg("n_neighbors"), py::arg("verbose") = false)
        .def("build_CEOs_Hash", &CEOs::build_CEOs_Hash, py::arg("dataset"))
        .def("search_CEOs_Hash", &CEOs::search_CEOs_Hash,
            py::arg("queries"), py::arg("n_neighbors"), py::arg("verbose") = false)
        ;

    // streamCEOs
    py::class_<streamCEOs>(m, "streamCEOs")
    .def(py::init<const int&>(),  py::arg("n_features"))
    //
    .def("setIndexParam", &streamCEOs::set_streamCEOsParam,
    py::arg("n_proj"), py::arg("n_repeats") = 1, py::arg("top_points") = 1,
    py::arg("n_threads") = -1, py::arg("random_seed") = -1)
//        .def_readwrite("n_threads", &CEOs::n_threads)
    .def_readwrite("n_probed_vectors", &streamCEOs::n_probed_vectors)
    .def_readwrite("n_probed_points", &streamCEOs::n_probed_points)
    .def_readwrite("n_cand", &streamCEOs::n_cand)
    .def("set_threads", &streamCEOs::set_threads, py::arg("n_threads"))
    .def("clear", &streamCEOs::clear)
    //
    .def("build", &streamCEOs::build, py::arg("dataset"))
    .def("update", &streamCEOs::update, py::arg("new_dataset"),py::arg("n_delPoints") = 0)
    .def("estimate_search", &streamCEOs::estimate_search,
    py::arg("queries"), py::arg("n_neighbors"), py::arg("verbose") = false)
    .def("hash_search", &streamCEOs::hash_search,
    py::arg("queries"), py::arg("n_neighbors"), py::arg("verbose") = false)
    //
    ;

} // namespace CEOs
} // namespace python
