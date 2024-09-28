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
        .def_readwrite("top_proj", &CEOs::top_proj)
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
        .def("build_coCEOs", &CEOs::build_coCEOs, py::arg("dataset"))
        .def("add_coCEOs", &CEOs::add_coCEOs, py::arg("new_dataset"))
        .def("search_coCEOs", &CEOs::search_coCEOs,
            py::arg("queries"), py::arg("n_neighbors"), py::arg("verbose") = false)
        ;

    // coCEOs
    py::class_<coCEOs>(m, "coCEOs")
    .def(py::init<const int&>(),  py::arg("n_features"))
    //
    .def("setIndexParam", &coCEOs::setIndexParam,
    py::arg("n_proj"), py::arg("n_repeats") = 1, py::arg("top_points") = 1,
    py::arg("n_threads") = -1, py::arg("random_seed") = -1
    )
//        .def_readwrite("n_threads", &CEOs::n_threads)
    .def_readwrite("top_proj", &coCEOs::top_proj)
    .def_readwrite("n_cand", &coCEOs::n_cand)
    .def("set_threads", &coCEOs::set_threads, py::arg("n_threads"))
    .def("clear", &coCEOs::clear)
    //
    .def("build", &coCEOs::build, py::arg("dataset"))
    .def("add_remove", &coCEOs::add_remove, py::arg("new_dataset"),py::arg("n_delPoints") = 0)
    .def("search", &coCEOs::search,
    py::arg("queries"), py::arg("n_neighbors"), py::arg("verbose") = false);

} // namespace CEOs
} // namespace python
