#include <CEOs.h>
#include <coCEOs.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace python {

namespace py = pybind11;

PYBIND11_MODULE(CEOs, m) { // Must be the same name with class CEOs
    py::class_<CEOs>(m, "CEOs")
        .def(py::init<const int&, const int&>(),  py::arg("n_points"), py::arg("n_features"))
        //
        .def("setIndexParam", &CEOs::set_CEOsParam,
            py::arg("n_proj"), py::arg("n_repeats") = 1,
            py::arg("n_threads") = -1, py::arg("random_seed") = -1
        )
//        .def_readwrite("n_threads", &CEOs::n_threads)
        .def_readwrite("n_probedVectors", &CEOs::n_probedVectors)
        .def_readwrite("n_probedPoints", &CEOs::n_probedPoints)
        .def_readwrite("n_cand", &CEOs::n_cand)
        .def("set_threads", &CEOs::set_threads, py::arg("n_threads"))
        .def("clear", &CEOs::clear)
        //
        .def("build", &CEOs::build_CEOs, py::arg("dataset"))
        .def("search", &CEOs::search_CEOs,
             py::arg("queries"), py::arg("n_neighbors"), py::arg("verbose") = false)
        //
        .def("set_coCEOsParam", &CEOs::set_coCEOsParam,
        py::arg("n_proj"), py::arg("n_repeats") = 1, py::arg("top_points") = 1,
        py::arg("n_threads") = -1, py::arg("random_seed") = -1
        )
        //
        .def("build_coCEOs_Est", &CEOs::build_coCEOs_Est, py::arg("dataset"))
        .def("search_coCEOs_Est", &CEOs::search_coCEOs_Est,
            py::arg("queries"), py::arg("n_neighbors"), py::arg("verbose") = false)
        .def("build_coCEOs_Hash", &CEOs::build_coCEOs_Hash, py::arg("dataset"))
        .def("search_coCEOs_Hash", &CEOs::search_coCEOs_Hash,
            py::arg("queries"), py::arg("n_neighbors"), py::arg("verbose") = false)
        ;

    // coCEOs
    py::class_<coCEOs>(m, "coCEOs")
    .def(py::init<const int&>(),  py::arg("n_features"))
    //
    .def("setIndexParam", &coCEOs::setIndexParam,
    py::arg("n_proj"), py::arg("n_repeats") = 1, py::arg("top_points") = 1,
    py::arg("n_threads") = -1, py::arg("random_seed") = -1, py::arg("centering") = true
    )
//        .def_readwrite("n_threads", &CEOs::n_threads)
    .def_readwrite("n_probedVectors", &coCEOs::n_probedVectors)
    .def_readwrite("n_probedPoints", &coCEOs::n_probedPoints)
    .def_readwrite("n_cand", &coCEOs::n_cand)
    .def("set_threads", &coCEOs::set_threads, py::arg("n_threads"))
    .def("clear", &coCEOs::clear)
    //
    .def("build", &coCEOs::build, py::arg("dataset"))
    .def("update", &coCEOs::update, py::arg("new_dataset"),py::arg("n_delPoints") = 0)
    .def("estimate_search", &coCEOs::estimate_search,
    py::arg("queries"), py::arg("n_neighbors"), py::arg("verbose") = false)
    .def("hash_search", &coCEOs::hash_search,
    py::arg("queries"), py::arg("n_neighbors"), py::arg("verbose") = false)
    //
    ;

} // namespace CEOs
} // namespace python
