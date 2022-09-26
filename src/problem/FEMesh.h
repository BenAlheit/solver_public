#ifndef SOLVER_FEMESH_H
#define SOLVER_FEMESH_H

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/base/mpi.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <mpi.h>


#include <boost/functional/hash.hpp>
#include <unordered_map>

#include <fstream>

#include "../materials/OutputFlags.h"


using namespace std;
using namespace dealii;

struct array_hash {
    template<class T1, long unsigned int n_elements>
    size_t operator()(const array<T1, n_elements> &p) const {
        return boost::hash_range(p.begin(), p.end());
    }
};

struct unordered_vector_hash {
    template<class T>
    size_t operator()(const vector<T> &v) const {
        size_t out = 0;
        for (const auto vi: v) {
            out = out ^ hash<T>()(vi);
        }
        return out;
    }

    template<class T>
    bool operator()(const vector<T> &this_v, const vector<T> &other_v) const {
        size_t this_hash = 0;
        for (const auto vi: this_v) {
            this_hash = this_hash ^ hash<T>()(vi);
        }
        size_t other_hash = 0;
        for (const auto vi: other_v) {
            other_hash = other_hash ^ hash<T>()(vi);
        }

        return other_hash != this_hash;
    }
};

template<unsigned int dim>
class InitializedFE {
public:
    InitializedFE(const vector<types::global_dof_index> local_dof_indices,
                  const vector<types::global_dof_index> scalar_dof_indices,
                  const vector<types::global_dof_index> tensor_dof_indices,
                  const vector<unsigned int> local_nodes,
                  const vector<array<unsigned int, dim>> local_node_to_dof,
                  const array<unsigned int, dim> range,
                  const vector<vector<double>> shape_fns,
                  const vector<vector<Tensor<1, dim>>> grad_shape_fns,
                  const vector<double> JxWs,
                  const typename DoFHandler<dim>::active_cell_iterator &cell_ptr,
                  vector<StateBase<dim> *> &states,
                  const FullMatrix<double> M_inv)
            : local_dof_indices(local_dof_indices), scalar_dof_indices(scalar_dof_indices),
              tensor_dof_indices(tensor_dof_indices), local_nodes(local_nodes),
              local_node_to_dof(local_node_to_dof), range(range), shape_fns(shape_fns), grad_shape_fns(grad_shape_fns),
              JxWs(JxWs), cell_ptr(cell_ptr), states(states), M_inv(M_inv) {};

    const vector<array<unsigned int, dim>> local_node_to_dof;
    const vector<unsigned int> local_nodes;
    const array<unsigned int, dim> range;
    const vector<types::global_dof_index> local_dof_indices;
    const vector<types::global_dof_index> scalar_dof_indices;
    const vector<types::global_dof_index> tensor_dof_indices;
    const vector<vector<double>> shape_fns;
    const vector<vector<Tensor<1, dim> > > grad_shape_fns;
    const vector<double> JxWs;
    const typename DoFHandler<dim>::active_cell_iterator cell_ptr;
    vector<StateBase<dim> *> states;
    const FullMatrix<double> M_inv;

    Tensor<2, dim> grad_field(unsigned int qp, const Vector<double> &dof_values) const {
        Tensor<2, dim> out;
        out = 0;
        vector<Tensor<1, dim>> grad_phis = grad_shape_fns.at(qp);

        for (const auto &i_node: local_nodes) {
            for (const auto &i_comp: range) {
                for (const auto &j_comp: range) {
                    out[i_comp][j_comp] += dof_values[local_node_to_dof.at(i_node).at(i_comp)]
                                           * grad_phis.at(i_node)[j_comp];
                }
            }
        }

        return out;
    }
};

template<unsigned int dim>
class FEMesh {
public:
    typedef vector<unsigned int> material_id_combination;

//    FEMesh(unsigned int order,
//           map<unsigned int, Material<dim> *> materials);
    FEMesh(unsigned int order,
           const map<unsigned int, Material<dim> *> &materials,
           const Triangulation<dim> &in_triangulation,
           const unsigned int &quadrature_dif = 1);

    void set_output_dir(string _output_dir) { output_dir = _output_dir; };

    void update_states();

    double get_JxW(unsigned int level, unsigned int el_id, unsigned int qp) const;

    vector<double> get_shape_fns(unsigned int level, unsigned int el_id, unsigned int qp) const;

    StateBase<dim> *get_state(unsigned int level, unsigned int el_id, unsigned int qp) const;

    vector<double> get_shape_fns(unsigned int qp) const;

    vector<Tensor<1, dim>> get_grad_shape_fns(unsigned int level, unsigned int el_id, unsigned int qp) const;

    template<class T>
    array<T, dim> grad_field(unsigned int level,
                             unsigned int el_id,
                             unsigned int qp,
                             vector<T> nodal_values) const;

    Tensor<2, dim> grad_field(unsigned int level,
                              unsigned int el_id,
                              unsigned int qp,
                              Vector<double> dof_values) const;

    Tensor<1, dim>
    get_grad_shape_fn(unsigned int level, unsigned int el_id, unsigned int qp, unsigned int i_node) const;

    const vector<InitializedFE<dim>> &get_locally_relevant_elements() const { return locally_relevant_elements; };

    const Triangulation<dim> &get_triangulation() const { return triangulation; };

    const FESystem<dim> &get_fe() const { return fe; };

    const DoFHandler<dim> &get_dof_handler() const { return dof_handler; };

    const DoFHandler<dim> &get_scalar_dof_handler() const { return scalar_dof_handler; };

    const DoFHandler<dim> &get_tensor_dof_handler() const { return tensor_dof_handler; };

    QGauss<dim> get_qgauss() const { return quadrature_formula; };

    array<unsigned int, dim> get_range() const { return range; };

    vector<unsigned int> get_qp_range() const { return qp_range; };

    vector<unsigned int> get_dof_range() const { return dof_range; };

    vector<unsigned int> get_local_nodes() const { return local_nodes; };

    vector<array<unsigned int, dim>> get_local_node_to_dof() const { return local_node_to_dof; };

    array<unsigned int, dim> get_local_node_to_dofs(unsigned int i_node) const { return local_node_to_dof.at(i_node); };

    const MPI_Comm &get_mpi_communicator() const { return mpi_communicator; };

    unsigned int get_n_mpi_processes() const { return n_mpi_processes; };

    unsigned int get_this_mpi_processes() const { return this_mpi_process; };

    ConditionalOStream get_pcout() const { return pcout; };

    unsigned int get_dofs_per_cell() const { return dofs_per_cell; };

    unsigned int get_n_q_points() const { return n_q_points; };

    unsigned int get_n_shape_fns() const { return n_shape_fns; };

    Material<dim> *get_material(unsigned int mat_id) { return materials.at(mat_id); };


    Vector<double>
    project_scalar_qp_field(const ScalarOutputFlag &flag, const vector<unsigned int> &material_ids) const;

    Vector<double>
    project_scalar_qp_field_el_wise(const ScalarOutputFlag &flag, const vector<unsigned int> &material_ids);

    Vector<double>
    project_n_scalar_qp_field(const nScalarOutputFlag &flag,
                              const vector<unsigned int> &material_ids,
                              const unsigned int &i) const;

    Vector<double>
    project_n_scalar_qp_field_el_wise(const nScalarOutputFlag &flag,
                                      const vector<unsigned int> &material_ids,
                                      const unsigned int &i);

    Vector<double>
    project_vector_qp_field(const VectorOutputFlag &flag, const vector<unsigned int> &material_ids) const;

    Vector<double>
    project_vector_qp_field_el_wise(const VectorOutputFlag &flag,
                                    const vector<unsigned int> &material_ids);

    Vector<double>
    project_n_vector_qp_field(const nVectorOutputFlag &flag,
                              const vector<unsigned int> &material_ids,
                              const unsigned int &i) const;

    Vector<double>
    project_n_vector_qp_field_el_wise(const nVectorOutputFlag &flag,
                                      const vector<unsigned int> &material_ids,
                                      const unsigned int &i);

    Vector<double>
    project_tensor_qp_field(const TensorOutputFlag &flag, const vector<unsigned int> &material_ids) const;

    Vector<double>
    project_tensor_qp_field_el_wise(const TensorOutputFlag &flag,
                                    const vector<unsigned int> &material_ids);

    Vector<double>
    project_n_tensor_qp_field(const nTensorOutputFlag &flag,
                              const vector<unsigned int> &material_ids,
                              const unsigned int &i) const;

    Vector<double>
    project_n_tensor_qp_field_el_wise(const nTensorOutputFlag &flag,
                                      const vector<unsigned int> &material_ids,
                                      const unsigned int &i);

    Vector<double>
    mesh_output_values(const MeshOutputFlag &flag) const;

    void output_averages(const Time *time,
                         const map<ScalarOutputFlag, vector<unsigned int>> &scalar_volume_average,
                         const map<VectorOutputFlag, vector<unsigned int>> &vector_volume_average,
                         const map<TensorOutputFlag, vector<unsigned int>> &tensor_volume_average) {
        output_scalar_averages(time, scalar_volume_average);
        output_vector_averages(time, vector_volume_average);
        output_tensor_averages(time, tensor_volume_average);
    };

    void output_scalar_averages(const Time *time,
                                const map<ScalarOutputFlag, vector<unsigned int>> &scalar_volume_average) const;

    void output_vector_averages(const Time *time,
                                const map<VectorOutputFlag, vector<unsigned int>> &vector_volume_average) const;

    void output_tensor_averages(const Time *time,
                                const map<TensorOutputFlag, vector<unsigned int>> &tensor_volume_average) const;


protected:

private:
    array<unsigned int, dim> range;
    vector<unsigned int> qp_range;
    vector<unsigned int> local_nodes;
    vector<unsigned int> dof_range;
    MPI_Comm mpi_communicator;

    string output_dir;

    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;
    ConditionalOStream pcout;

    Triangulation<dim> triangulation;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;
    QGauss<dim> quadrature_formula;

    DoFHandler<dim> projection_dof_handler;
    SparseMatrix<double> projection_matrix;
    SparseDirectUMFPACK projection_solver;

    DoFHandler<dim> scalar_dof_handler;
    DoFHandler<dim> tensor_dof_handler;

    vector<array<unsigned int, dim>> local_node_to_dof;
    map<unsigned int, Material<dim> *> materials;

    vector<vector<double>> shape_fns; // @ qp -> node array
    unordered_map<array<unsigned int, 3>, vector<Tensor<1, dim>>, array_hash>
            grad_shape_fns; // @ level, el_id, qp -> node array
    unordered_map<array<unsigned int, 3>, double, array_hash> JxWs; // @ level, el_id, qp -> node array
    unordered_map<array<unsigned int, 3>, StateBase<dim> *, array_hash>
            states; // @ level, el_id, qp -> material state

    vector<InitializedFE<dim>> locally_relevant_elements;

    unordered_map<material_id_combination, PETScWrappers::MPI::Vector, unordered_vector_hash> n_connected_elements_material_combo;

    PETScWrappers::MPI::Vector projected_scalar_values;
    PETScWrappers::MPI::Vector projected_vector_values;
    PETScWrappers::MPI::Vector projected_tensor_values;

    unsigned int dofs_per_cell;
    unsigned int n_q_points;
    unsigned int n_shape_fns;


    void init_maps();

    void init_local_elements();

    void init_projection_matrix();

    Vector<double>
    material_id_output() const;

    Vector<double>
    boundary_id_output() const;

    void add_material_combo_connected_elements(const material_id_combination &mat_combo);

};

template<unsigned int dim>
FEMesh<dim>::FEMesh(unsigned int order,
                    const map<unsigned int, Material<dim> *> &materials,
                    const Triangulation<dim> &in_triangulation,
                    const unsigned int &quadrature_dif)
        : mpi_communicator(MPI_COMM_WORLD), n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
          this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)), pcout(cout, (this_mpi_process == 0)),
          fe(FE_Q<dim>(order), dim)
//        , triangulation(in_triangulation)
        , dof_handler(triangulation), quadrature_formula(fe.degree + quadrature_dif), materials(materials),
          projection_dof_handler(triangulation), scalar_dof_handler(triangulation), tensor_dof_handler(triangulation) {
    pcout << "n mpi processes: " << n_mpi_processes << endl;

    triangulation.copy_triangulation(in_triangulation);
    dof_handler.reinit(triangulation);
    iota(range.begin(), range.end(), 0);


    GridTools::partition_triangulation(n_mpi_processes, triangulation);
    dof_handler.distribute_dofs(fe);
    DoFRenumbering::subdomain_wise(dof_handler);
    projected_vector_values.reinit(DoFTools::locally_owned_dofs_per_subdomain(dof_handler)[this_mpi_process],
                                   mpi_communicator);

    scalar_dof_handler.initialize(triangulation, fe.base_element(0));
    scalar_dof_handler.distribute_dofs(fe.base_element(0));
    DoFRenumbering::subdomain_wise(scalar_dof_handler);
    projected_scalar_values.reinit(DoFTools::locally_owned_dofs_per_subdomain(scalar_dof_handler)[this_mpi_process],
                                   mpi_communicator);

    vector<types::global_dof_index> scalar_dof_indices(fe.base_element(0).n_dofs_per_cell());


    FESystem<dim, dim> tensor_fe(FE_Q<dim, dim>(fe.degree), dim * dim);
    vector<types::global_dof_index> tensor_dof_indices(tensor_fe.n_dofs_per_cell());
    tensor_dof_handler.initialize(triangulation, tensor_fe);
    DoFRenumbering::subdomain_wise(tensor_dof_handler);
    projected_tensor_values.reinit(DoFTools::locally_owned_dofs_per_subdomain(tensor_dof_handler)[this_mpi_process],
                                   mpi_communicator);


    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    dofs_per_cell = fe.n_dofs_per_cell();
    n_q_points = quadrature_formula.size();
    n_shape_fns = fe_values.get_fe().base_element(0).n_dofs_per_cell();

    local_nodes = vector<unsigned int>(n_shape_fns);
    iota(local_nodes.begin(), local_nodes.end(), 0);

    local_node_to_dof = vector<array<unsigned int, dim >>(n_shape_fns);

    for (auto const &i_node: local_nodes)
        for (auto const &i_comp: range)
            local_node_to_dof.at(i_node).at(i_comp) = fe_values.get_fe().component_to_system_index(i_comp, i_node);

    vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    dof_range = vector<unsigned int>(dofs_per_cell);
    iota(dof_range.begin(), dof_range.end(), 0);


    vector<Tensor<1, dim>> el_grad_shape_fns(n_shape_fns);
    shape_fns = vector<vector<double >>(n_q_points, vector<double>(n_shape_fns));
    qp_range = vector<unsigned int>(n_q_points);
    iota(qp_range.begin(), qp_range.end(), 0);

    for (const auto &cell: dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        for (const unsigned int q_point: qp_range) {
            for (const auto &i_node: local_nodes) {
                shape_fns.at(q_point).at(i_node) = fe_values.shape_value(local_node_to_dof.at(i_node).at(0),
                                                                         q_point);
            }
        }
        break;
    }

//    init_maps();
    init_local_elements();
//    init_projection_matrix();

//    vector<vector<Tensor<1, dim>>> el_grad_shape_fns_at_qp(n_q_points, vector<Tensor<1, dim>>(n_shape_fns));
//    vector<double> el_JxWs(n_q_points);
//    vector<StateBase<dim> *> el_states(n_q_points);
//    FullMatrix<double> M(n_shape_fns, n_shape_fns), M_inv(n_shape_fns, n_shape_fns);
//
//
//    for (const auto &cell: dof_handler.active_cell_iterators()) {
////        if (cell->subdomain_id() == this_mpi_process) {
//        if (true) {
//            M = 0.0;
//            fe_values.reinit(cell);
//            for (const unsigned int q_point: qp_range) {
//                for (const auto &i_node: local_nodes) {
//                    el_grad_shape_fns.at(i_node) = fe_values.shape_grad(local_node_to_dof.at(i_node).at(0), q_point);
//                }
//
//                el_grad_shape_fns_at_qp.at(q_point) = el_grad_shape_fns;
//                el_JxWs.at(q_point) = fe_values.JxW(q_point);
//                el_states.at(q_point) = materials.at(cell->material_id())->create_state();
//
//                grad_shape_fns[{cell->level(), cell->index(), q_point}] = el_grad_shape_fns_at_qp.at(q_point);
//                JxWs[{cell->level(), cell->index(), q_point}] = el_JxWs.at(q_point);
//                states[{cell->level(), cell->index(), q_point}] = el_states.at(q_point);
//            }
//        }
//        for (const unsigned int q_point: qp_range)
//            for (const auto &i_node: local_nodes)
//                for (const auto &j_node: local_nodes)
//                    M[i_node][j_node] += shape_fns.at(q_point).at(i_node)
//                                         * shape_fns.at(q_point).at(j_node)
//                                         * el_JxWs.at(q_point);
//
//        M_inv.invert(M);
//
//        if (cell->subdomain_id() == this_mpi_process) {
//            cell->get_dof_indices(local_dof_indices);
//            const DoFCellAccessor<dim, dim, false> salar_dof_accessor(&triangulation,
//                                                                      cell->level(),
//                                                                      cell->index(),
//                                                                      &scalar_dof_handler);
//            salar_dof_accessor.get_dof_indices(scalar_dof_indices);
//            const DoFCellAccessor<dim, dim, false> tensor_dof_accessor(&triangulation,
//                                                                       cell->level(),
//                                                                       cell->index(),
//                                                                       &tensor_dof_handler);
//            tensor_dof_accessor.get_dof_indices(tensor_dof_indices);
//
//            locally_relevant_elements.push_back(InitializedFE<dim>(local_dof_indices,
//                                                                   scalar_dof_indices,
//                                                                   tensor_dof_indices,
//                                                                   local_nodes,
//                                                                   local_node_to_dof,
//                                                                   range,
//                                                                   shape_fns,
//                                                                   el_grad_shape_fns_at_qp,
//                                                                   el_JxWs,
//                                                                   cell,
//                                                                   el_states,
//                                                                   M_inv));
//        }
//    }

//    projection_dof_handler.initialize(triangulation, fe.base_element(0));
//    projection_dof_handler.distribute_dofs(fe.base_element(0));
//
//    DynamicSparsityPattern dsp(projection_dof_handler.n_dofs());
//    DoFTools::make_sparsity_pattern(projection_dof_handler, dsp);
//    SparsityPattern sparsity_pattern;
//    sparsity_pattern.copy_from(dsp);
//    projection_matrix.reinit(sparsity_pattern);
//    vector<types::global_dof_index> projection_local_dof_indices(n_shape_fns);
//    double JxW;
//
//
//    for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
////        fe_values.reinit(cell);
//        cell->get_dof_indices(projection_local_dof_indices);
//
//        for (const auto &qp: qp_range) {
//            JxW = get_JxW(cell->level(), cell->index(), qp);
//            for (const auto &i: local_nodes)
//                for (const auto &j: local_nodes)
//                    projection_matrix.add(projection_local_dof_indices[i],
//                                          projection_local_dof_indices[j],
//                                          shape_fns.at(qp).at(i) * shape_fns.at(qp).at(j) * JxW);
//        }
//    }
//
//    projection_solver.template initialize(projection_matrix);


}

template<unsigned int dim>
void FEMesh<dim>::init_local_elements() {

    vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    vector<types::global_dof_index> scalar_dof_indices(fe.base_element(0).n_dofs_per_cell());
    FESystem<dim, dim> tensor_fe(FE_Q<dim, dim>(fe.degree), dim * dim);
    vector<types::global_dof_index> tensor_dof_indices(tensor_fe.n_dofs_per_cell());

    dof_range = vector<unsigned int>(dofs_per_cell);
    iota(dof_range.begin(), dof_range.end(), 0);

    vector<Tensor<1, dim>> el_grad_shape_fns(n_shape_fns);
    vector<vector<Tensor<1, dim>>> el_grad_shape_fns_at_qp(n_q_points, vector<Tensor<1, dim>>(n_shape_fns));
    vector<double> el_JxWs(n_q_points);
    vector<StateBase<dim> *> el_states(n_q_points);
    FullMatrix<double> M(n_shape_fns, n_shape_fns), M_inv(n_shape_fns, n_shape_fns);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    for (const auto &cell: dof_handler.active_cell_iterators()) {
        if (cell->subdomain_id() == this_mpi_process) {
            fe_values.reinit(cell);
            for (const unsigned int q_point: qp_range) {
                for (const auto &i_node: local_nodes) {
                    el_grad_shape_fns.at(i_node) = fe_values.shape_grad(local_node_to_dof.at(i_node).at(0), q_point);
                }

                el_grad_shape_fns_at_qp.at(q_point) = el_grad_shape_fns;
                el_JxWs.at(q_point) = fe_values.JxW(q_point);
                el_states.at(q_point) = materials.at(cell->material_id())->create_state();

            }

            M = 0.0;
            for (const unsigned int q_point: qp_range)
                for (const auto &i_node: local_nodes)
                    for (const auto &j_node: local_nodes)
                        M[i_node][j_node] += shape_fns.at(q_point).at(i_node)
                                             * shape_fns.at(q_point).at(j_node)
                                             * el_JxWs.at(q_point);

            M_inv.invert(M);


            const DoFCellAccessor<dim, dim, false> salar_dof_accessor(&triangulation,
                                                                      cell->level(),
                                                                      cell->index(),
                                                                      &scalar_dof_handler);
            salar_dof_accessor.get_dof_indices(scalar_dof_indices);

            cell->get_dof_indices(local_dof_indices);

            const DoFCellAccessor<dim, dim, false> tensor_dof_accessor(&triangulation,
                                                                       cell->level(),
                                                                       cell->index(),
                                                                       &tensor_dof_handler);
            tensor_dof_accessor.get_dof_indices(tensor_dof_indices);

            locally_relevant_elements.push_back(InitializedFE<dim>(local_dof_indices,
                                                                   scalar_dof_indices,
                                                                   tensor_dof_indices,
                                                                   local_nodes,
                                                                   local_node_to_dof,
                                                                   range,
                                                                   shape_fns,
                                                                   el_grad_shape_fns_at_qp,
                                                                   el_JxWs,
                                                                   cell,
                                                                   el_states,
                                                                   M_inv));
        }
    }
}

template<unsigned int dim>
void FEMesh<dim>::init_projection_matrix() {

    projection_dof_handler.initialize(triangulation, fe.base_element(0));
    projection_dof_handler.distribute_dofs(fe.base_element(0));

    DynamicSparsityPattern dsp(projection_dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(projection_dof_handler, dsp);
    SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dsp);
    projection_matrix.reinit(sparsity_pattern);
    vector<types::global_dof_index> projection_local_dof_indices(n_shape_fns);
    double JxW;

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);


    for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);

        cell->get_dof_indices(projection_local_dof_indices);

        for (const auto &qp: qp_range) {
            JxW = fe_values.JxW(qp);;
            for (const auto &i: local_nodes)
                for (const auto &j: local_nodes)
                    projection_matrix.add(projection_local_dof_indices[i],
                                          projection_local_dof_indices[j],
                                          shape_fns.at(qp).at(i) * shape_fns.at(qp).at(j) * JxW);
        }
    }

    projection_solver.template initialize(projection_matrix);

}

template<unsigned int dim>
void FEMesh<dim>::init_maps() {
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
    vector<Tensor<1, dim>> el_grad_shape_fns(n_shape_fns);

    for (const auto &cell: dof_handler.active_cell_iterators()) {
        if (cell->subdomain_id() == this_mpi_process) {
            fe_values.reinit(cell);
            for (const unsigned int q_point: qp_range) {
                for (const auto &i_node: local_nodes) {
                    el_grad_shape_fns.at(i_node) = fe_values.shape_grad(local_node_to_dof.at(i_node).at(0), q_point);
                }

                grad_shape_fns[{cell->level(), cell->index(), q_point}] = el_grad_shape_fns;
                JxWs[{cell->level(), cell->index(), q_point}] = fe_values.JxW(q_point);
                states[{cell->level(), cell->index(), q_point}] = materials.at(cell->material_id())->create_state();
            }
        }
    }
}

template<unsigned int dim>
double FEMesh<dim>::get_JxW(unsigned int level, unsigned int el_id, unsigned int qp) const {
    return JxWs.at(array<unsigned int, 3>({level, el_id, qp}));
}

template<unsigned int dim>
vector<double> FEMesh<dim>::get_shape_fns(unsigned int level, unsigned int el_id, unsigned int qp) const {
    return get_shape_fns(qp);
}

template<unsigned int dim>
vector<double> FEMesh<dim>::get_shape_fns(unsigned int qp) const {
    return shape_fns.at(qp);
}

template<unsigned int dim>
vector<Tensor<1, dim>> FEMesh<dim>::get_grad_shape_fns(unsigned int level, unsigned int el_id, unsigned int qp) const {
    return grad_shape_fns.at(array<unsigned int, 3>({level, el_id, qp}));
}

template<unsigned int dim>
Tensor<1, dim> FEMesh<dim>::get_grad_shape_fn(unsigned int level,
                                              unsigned int el_id,
                                              unsigned int qp,
                                              unsigned int i_node) const {
    return grad_shape_fns.at(array<unsigned int, 3>({level, el_id, qp})).at(i_node);
}

template<unsigned int dim>
StateBase<dim> *FEMesh<dim>::get_state(unsigned int level, unsigned int el_id, unsigned int qp) const {
    return states.at(array<unsigned int, 3>({level, el_id, qp}));
}

template<unsigned int dim>
template<class T>
array<T, dim>
FEMesh<dim>::grad_field(unsigned int level, unsigned int el_id, unsigned int qp, vector<T> nodal_values) const {
    array<T, dim> out;
    for (const auto &i_comp: range) out.at(i_comp) = 0;

    vector<Tensor<1, dim>> grad_phis = grad_shape_fns.at(array<unsigned int, 3>({level, el_id, qp}));

    for (const auto &i_node: local_nodes)
        for (const auto &i_comp: range)
            out.at(i_comp) += nodal_values.at(i_node) * grad_phis.at(i_node)[i_comp];

    return out;
}

template<unsigned int dim>
Tensor<2, dim>
FEMesh<dim>::grad_field(unsigned int level, unsigned int el_id, unsigned int qp, Vector<double> dof_values) const {
    Tensor<2, dim> out;
    out = 0;
    vector<Tensor<1, dim>> grad_phis = grad_shape_fns.at(array<unsigned int, 3>({level, el_id, qp}));

    for (const auto &i_node: local_nodes) {
        for (const auto &i_comp: range) {
            for (const auto &j_comp: range) {
                out[i_comp][j_comp] += dof_values[local_node_to_dof.at(i_node).at(i_comp)]
                                       * grad_phis.at(i_node)[j_comp];
            }
        }
    }

    return out;
}

template<unsigned int dim>
Vector<double>
FEMesh<dim>::project_scalar_qp_field(const ScalarOutputFlag &flag, const vector<unsigned int> &material_ids) const {
    Vector<double> projected_values;
    projected_values.reinit(projection_dof_handler.n_dofs());

    vector<types::global_dof_index> local_dof_indices(n_shape_fns);
    double JxW, state_val;
    StateBase<dim> *state;
    for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) != material_ids.end()) {

            cell->get_dof_indices(local_dof_indices);

            for (const auto &qp: qp_range) {
                JxW = get_JxW(cell->level(), cell->index(), qp);
                state = get_state(cell->level(), cell->index(), qp);
                state_val = state->scalar_output(flag);
                for (const auto &i: local_nodes)
                    projected_values[local_dof_indices[i]] += state_val * get_shape_fns(qp).at(i) * JxW;
            }
        }
    }


    projection_solver.solve(projected_values);


    bool neighbour_is_not_of_correct_material = true;
    for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) == material_ids.end()) {

            for (unsigned int i_face = 0; i_face < cell->n_faces(); i_face++) {
                try {
                    neighbour_is_not_of_correct_material =
                            find(material_ids.begin(), material_ids.end(), cell->neighbor(i_face)->material_id()) ==
                            material_ids.end();
                } catch (...) {}
                if (not neighbour_is_not_of_correct_material)
                    break;
            }

            if (neighbour_is_not_of_correct_material) {
                cell->get_dof_indices(local_dof_indices);


                for (const auto &dof_index: local_dof_indices)
                    projected_values[dof_index] = nan("");
            } else {
                neighbour_is_not_of_correct_material = true;
            }
        }
    }

//    for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
//        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) == material_ids.end()) {
//
//            cell->get_dof_indices(local_dof_indices);
//            for (const auto &dof_index: local_dof_indices)
//                projected_values[dof_index] = nan("");
//        }
//    }

    return projected_values;
}

template<unsigned int dim>
Vector<double>
FEMesh<dim>::project_scalar_qp_field_el_wise(const ScalarOutputFlag &flag,
                                             const vector<unsigned int> &material_ids) {
    if (n_connected_elements_material_combo.find(material_ids) == n_connected_elements_material_combo.end()) {
        add_material_combo_connected_elements(material_ids);
    }

    Vector<double> connected_elements(n_connected_elements_material_combo[material_ids]);

//    Vector<double> projected_values;
//    projected_values.reinit(projection_dof_handler.n_dofs());
    PETScWrappers::MPI::Vector projected_values;
    projected_values.reinit(DoFTools::locally_owned_dofs_per_subdomain(scalar_dof_handler)[this_mpi_process],
                            mpi_communicator);

    Vector<double> local_rhs(n_shape_fns), local_projection(n_shape_fns);

    vector<types::global_dof_index> local_dof_indices(n_shape_fns);
    double JxW, state_val;
    StateBase<dim> *state;


    for (const auto &cell: locally_relevant_elements) {
        if (find(material_ids.begin(), material_ids.end(), cell.cell_ptr->material_id()) != material_ids.end()) {
            local_rhs = 0.0;
            local_projection = 0.0;

            for (const auto &qp: qp_range) {
                JxW = cell.JxWs.at(qp);
                state = cell.states.at(qp);
                state_val = state->scalar_output(flag);
                for (const auto &i: local_nodes)
                    local_rhs[i] += state_val * get_shape_fns(qp).at(i) * JxW;
            }
            cell.M_inv.template vmult(local_projection, local_rhs);

            for (const auto &i_node: local_nodes)
                projected_values[cell.scalar_dof_indices.at(i_node)] += local_projection[i_node]
                                                                        / connected_elements[cell.scalar_dof_indices.at(
                        i_node)];
        }
    }


    projected_values.compress(VectorOperation::add);

    Vector<double> out(projected_values);

    bool neighbour_is_not_of_correct_material = true;
    for (const auto &cell: scalar_dof_handler.active_cell_iterators()) {
        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) == material_ids.end()) {

            for (unsigned int i_face = 0; i_face < cell->n_faces(); i_face++) {
                try {
                    neighbour_is_not_of_correct_material =
                            find(material_ids.begin(), material_ids.end(), cell->neighbor(i_face)->material_id()) ==
                            material_ids.end();
                } catch (...) {}
                if (not neighbour_is_not_of_correct_material)
                    break;
            }

            if (neighbour_is_not_of_correct_material) {
                cell->get_dof_indices(local_dof_indices);


                for (const auto &dof_index: local_dof_indices)
                    out[dof_index] = nan("");
            } else {
                neighbour_is_not_of_correct_material = true;
            }
        }
    }

    return out;
}


template<unsigned int dim>
Vector<double>
FEMesh<dim>::project_n_scalar_qp_field(const nScalarOutputFlag &flag,
                                       const vector<unsigned int> &material_ids,
                                       const unsigned int &i) const {
    Vector<double> projected_values;
    projected_values.reinit(projection_dof_handler.n_dofs());

    vector<types::global_dof_index> local_dof_indices(n_shape_fns);
    double JxW, state_val;
    StateBase<dim> *state;
    for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) != material_ids.end()) {

            cell->get_dof_indices(local_dof_indices);

            for (const auto &qp: qp_range) {
                JxW = get_JxW(cell->level(), cell->index(), qp);
                state = get_state(cell->level(), cell->index(), qp);
//                state_val = state->scalar_output(flag);
                state_val = state->n_scalar_output(flag, i);
                for (const auto &i: local_nodes)
                    projected_values[local_dof_indices[i]] += state_val * get_shape_fns(qp).at(i) * JxW;
            }
        }
    }


    projection_solver.solve(projected_values);


    bool neighbour_is_not_of_correct_material = true;
    for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) == material_ids.end()) {

            for (unsigned int i_face = 0; i_face < cell->n_faces(); i_face++) {
                try {
                    neighbour_is_not_of_correct_material =
                            find(material_ids.begin(), material_ids.end(), cell->neighbor(i_face)->material_id()) ==
                            material_ids.end();
                } catch (...) {}
                if (not neighbour_is_not_of_correct_material)
                    break;
            }

            if (neighbour_is_not_of_correct_material) {
                cell->get_dof_indices(local_dof_indices);


                for (const auto &dof_index: local_dof_indices)
                    projected_values[dof_index] = nan("");
            } else {
                neighbour_is_not_of_correct_material = true;
            }
        }
    }

//    for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
//        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) == material_ids.end()) {
//
//            cell->get_dof_indices(local_dof_indices);
//            for (const auto &dof_index: local_dof_indices)
//                projected_values[dof_index] = nan("");
//        }
//    }

    return projected_values;
}

template<unsigned int dim>
Vector<double>
FEMesh<dim>::project_n_scalar_qp_field_el_wise(const nScalarOutputFlag &flag,
                                               const vector<unsigned int> &material_ids,
                                               const unsigned int &i) {
    if (n_connected_elements_material_combo.find(material_ids) == n_connected_elements_material_combo.end()) {
        add_material_combo_connected_elements(material_ids);
    }

    Vector<double> connected_elements(n_connected_elements_material_combo.at(material_ids));

//    Vector<double> projected_values;
//    projected_values.reinit(projection_dof_handler.n_dofs());
    PETScWrappers::MPI::Vector projected_values;
    projected_values.reinit(DoFTools::locally_owned_dofs_per_subdomain(scalar_dof_handler)[this_mpi_process],
                            mpi_communicator);

    Vector<double> local_rhs(n_shape_fns), local_projection(n_shape_fns);

    vector<types::global_dof_index> local_dof_indices(n_shape_fns);
    double JxW, state_val;
    StateBase<dim> *state;


    for (const auto &cell: locally_relevant_elements) {
        if (find(material_ids.begin(), material_ids.end(), cell.cell_ptr->material_id()) != material_ids.end()) {
            local_rhs = 0.0;
            local_projection = 0.0;

            for (const auto &qp: qp_range) {
                JxW = cell.JxWs.at(qp);
                state = cell.states.at(qp);
                state_val = state->n_scalar_output(flag, i);

                for (const auto &i_node: local_nodes)
                    local_rhs[i_node] += state_val * get_shape_fns(qp).at(i_node) * JxW;
            }
            cell.M_inv.template vmult(local_projection, local_rhs);

            for (const auto &i_node: local_nodes)
                projected_values[cell.scalar_dof_indices.at(i_node)] += local_projection[i_node]
                                                                        / connected_elements[cell.scalar_dof_indices.at(
                        i_node)];
        }
    }

    bool neighbour_is_not_of_correct_material = true;


    projected_values.compress(VectorOperation::add);

    Vector<double> out(projected_values);

    for (const auto &cell: scalar_dof_handler.active_cell_iterators()) {
        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) == material_ids.end()) {

            for (unsigned int i_face = 0; i_face < cell->n_faces(); i_face++) {
                try {
                    neighbour_is_not_of_correct_material =
                            find(material_ids.begin(), material_ids.end(), cell->neighbor(i_face)->material_id()) ==
                            material_ids.end();
                } catch (...) {}
                if (not neighbour_is_not_of_correct_material)
                    break;
            }

            if (neighbour_is_not_of_correct_material) {
                cell->get_dof_indices(local_dof_indices);


                for (const auto &dof_index: local_dof_indices)
                    out[dof_index] = nan("");
            } else {
                neighbour_is_not_of_correct_material = true;
            }
        }
    }
    return out;
}

template<unsigned int dim>
Vector<double>
FEMesh<dim>::project_vector_qp_field(const VectorOutputFlag &flag, const vector<unsigned int> &material_ids) const {
    Vector<double> projected_values;
    projected_values.reinit(dof_handler.n_dofs());

    vector<unsigned int> projection_dof_range(projection_dof_handler.n_dofs());
    iota(projection_dof_range.begin(), projection_dof_range.end(), 0);

    vector<Vector<double>> comp_wise_values(dim);
    for (const auto &i_comp: range)
        comp_wise_values.at(i_comp).reinit(projection_dof_handler.n_dofs());

    vector<types::global_dof_index> local_dof_indices(n_shape_fns);
    double JxW;
    Tensor<1, dim> state_val;
    StateBase<dim> *state;
    for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) != material_ids.end()) {

            cell->get_dof_indices(local_dof_indices);

            for (const auto &qp: qp_range) {
                JxW = get_JxW(cell->level(), cell->index(), qp);
                state = get_state(cell->level(), cell->index(), qp);
                state_val = state->vector_output(flag);
                for (const auto &i: local_nodes)
                    for (const auto &i_comp: range)
                        comp_wise_values.at(i_comp)[local_dof_indices[i]] +=
                                state_val[i_comp] * get_shape_fns(qp).at(i) * JxW;
            }
        }
    }

    for (const auto &i_comp: range)
        projection_solver.solve(comp_wise_values.at(i_comp));

//    for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
//        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) == material_ids.end()) {
//
//            cell->get_dof_indices(local_dof_indices);
//            for (const auto &dof_index: local_dof_indices)
//                for (const auto &i_comp: range)
//                    comp_wise_values.at(i_comp)[dof_index] = nan("");
//        }
//    }

    bool neighbour_is_not_of_correct_material = true;
    for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) == material_ids.end()) {

            for (unsigned int i_face = 0; i_face < cell->n_faces(); i_face++) {
                try {
                    neighbour_is_not_of_correct_material =
                            find(material_ids.begin(), material_ids.end(), cell->neighbor(i_face)->material_id()) ==
                            material_ids.end();
                } catch (...) {}
                if (not neighbour_is_not_of_correct_material)
                    break;
            }

            if (neighbour_is_not_of_correct_material) {
                cell->get_dof_indices(local_dof_indices);
                for (const auto &dof_index: local_dof_indices)
                    for (const auto &i_comp: range)
                        comp_wise_values.at(i_comp)[dof_index] = nan("");
            } else {
                neighbour_is_not_of_correct_material = true;
            }
        }
    }

    for (const auto &i_comp: range)
        for (const auto &k: projection_dof_range)
            projected_values[k * dim + i_comp] = comp_wise_values.at(i_comp)[k];

    return projected_values;
}

template<unsigned int dim>
Vector<double>
FEMesh<dim>::project_vector_qp_field_el_wise(const VectorOutputFlag &flag,
                                             const vector<unsigned int> &material_ids) {
    if (n_connected_elements_material_combo.find(material_ids) == n_connected_elements_material_combo.end()) {
        add_material_combo_connected_elements(material_ids);
    }

    Vector<double> connected_elements(n_connected_elements_material_combo[material_ids]);

//    Vector<double> projected_values;
//    projected_values.reinit(dof_handler.n_dofs());

    PETScWrappers::MPI::Vector projected_values;
    projected_values.reinit(DoFTools::locally_owned_dofs_per_subdomain(dof_handler)[this_mpi_process],
                            mpi_communicator);

    Vector<double> local_rhs(n_shape_fns), local_projection(n_shape_fns);

    vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    double JxW;
    Tensor<1, dim> state_val;
    StateBase<dim> *state;

    for (const auto &cell: locally_relevant_elements) {
        if (find(material_ids.begin(), material_ids.end(), cell.cell_ptr->material_id()) != material_ids.end()) {
            for (const auto &i_comp: range) {
                local_rhs = 0.0;
                local_projection = 0.0;
                for (const auto &qp: qp_range) {
                    JxW = cell.JxWs.at(qp);
                    state = cell.states.at(qp);
                    state_val = state->vector_output(flag);
                    for (const auto &i: local_nodes)
                        local_rhs[i] += state_val[i_comp] * get_shape_fns(qp).at(i) * JxW;
                }
                cell.M_inv.template vmult(local_projection, local_rhs);

                for (const auto &i_node: local_nodes)
                    projected_values[cell.local_dof_indices.at(cell.local_node_to_dof.at(i_node).at(i_comp))]
                            += local_projection[i_node] / connected_elements[cell.scalar_dof_indices.at(i_node)];
            }
        }
    }

    projected_values.compress(VectorOperation::add);

    Vector<double> out(projected_values);

    bool neighbour_is_not_of_correct_material = true;
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) == material_ids.end()) {

            for (unsigned int i_face = 0; i_face < cell->n_faces(); i_face++) {
                try {
                    neighbour_is_not_of_correct_material =
                            find(material_ids.begin(), material_ids.end(), cell->neighbor(i_face)->material_id()) ==
                            material_ids.end();
                } catch (...) {}
                if (not neighbour_is_not_of_correct_material)
                    break;
            }

            if (neighbour_is_not_of_correct_material) {
                cell->get_dof_indices(local_dof_indices);


                for (const auto &dof_index: local_dof_indices)
                    out[dof_index] = nan("");
            } else {
                neighbour_is_not_of_correct_material = true;
            }
        }
    }

    return out;
}


template<unsigned int dim>
Vector<double>
FEMesh<dim>::project_n_vector_qp_field(const nVectorOutputFlag &flag,
                                       const vector<unsigned int> &material_ids,
                                       const unsigned int &i) const {
    Vector<double> projected_values;
    projected_values.reinit(dof_handler.n_dofs());

    vector<unsigned int> projection_dof_range(projection_dof_handler.n_dofs());
    iota(projection_dof_range.begin(), projection_dof_range.end(), 0);

    vector<Vector<double>> comp_wise_values(dim);
    for (const auto &i_comp: range)
        comp_wise_values.at(i_comp).reinit(projection_dof_handler.n_dofs());

    vector<types::global_dof_index> local_dof_indices(n_shape_fns);
    double JxW;
    Tensor<1, dim> state_val;
    StateBase<dim> *state;
    for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) != material_ids.end()) {

            cell->get_dof_indices(local_dof_indices);

            for (const auto &qp: qp_range) {
                JxW = get_JxW(cell->level(), cell->index(), qp);
                state = get_state(cell->level(), cell->index(), qp);
//                state_val = state->vector_output(flag);
                state_val = state->n_vector_output(flag, i);
                for (const auto &i: local_nodes)
                    for (const auto &i_comp: range)
                        comp_wise_values.at(i_comp)[local_dof_indices[i]] +=
                                state_val[i_comp] * get_shape_fns(qp).at(i) * JxW;
            }
        }
    }

    for (const auto &i_comp: range)
        projection_solver.solve(comp_wise_values.at(i_comp));

//    for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
//        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) == material_ids.end()) {
//
//            cell->get_dof_indices(local_dof_indices);
//            for (const auto &dof_index: local_dof_indices)
//                for (const auto &i_comp: range)
//                    comp_wise_values.at(i_comp)[dof_index] = nan("");
//        }
//    }

    bool neighbour_is_not_of_correct_material = true;
    for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) == material_ids.end()) {

            for (unsigned int i_face = 0; i_face < cell->n_faces(); i_face++) {
                try {
                    neighbour_is_not_of_correct_material =
                            find(material_ids.begin(), material_ids.end(), cell->neighbor(i_face)->material_id()) ==
                            material_ids.end();
                } catch (...) {}
                if (not neighbour_is_not_of_correct_material)
                    break;
            }

            if (neighbour_is_not_of_correct_material) {
                cell->get_dof_indices(local_dof_indices);
                for (const auto &dof_index: local_dof_indices)
                    for (const auto &i_comp: range)
                        comp_wise_values.at(i_comp)[dof_index] = nan("");
            } else {
                neighbour_is_not_of_correct_material = true;
            }
        }
    }

    for (const auto &i_comp: range)
        for (const auto &k: projection_dof_range)
            projected_values[k * dim + i_comp] = comp_wise_values.at(i_comp)[k];

    return projected_values;
}


template<unsigned int dim>
Vector<double>
FEMesh<dim>::project_n_vector_qp_field_el_wise(const nVectorOutputFlag &flag,
                                               const vector<unsigned int> &material_ids,
                                               const unsigned int &i) {
    if (n_connected_elements_material_combo.find(material_ids) == n_connected_elements_material_combo.end()) {
        add_material_combo_connected_elements(material_ids);
    }

    Vector<double> connected_elements(n_connected_elements_material_combo[material_ids]);

//    Vector<double> projected_values;
//    projected_values.reinit(dof_handler.n_dofs());

    PETScWrappers::MPI::Vector projected_values;
    projected_values.reinit(DoFTools::locally_owned_dofs_per_subdomain(dof_handler)[this_mpi_process],
                            mpi_communicator);

    Vector<double> local_rhs(n_shape_fns), local_projection(n_shape_fns);

    vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    double JxW;
    Tensor<1, dim> state_val;
    StateBase<dim> *state;

    for (const auto &cell: locally_relevant_elements) {
        if (find(material_ids.begin(), material_ids.end(), cell.cell_ptr->material_id()) != material_ids.end()) {
            for (const auto &i_comp: range) {
                local_rhs = 0.0;
                local_projection = 0.0;
                for (const auto &qp: qp_range) {
                    JxW = cell.JxWs.at(qp);
                    state = cell.states.at(qp);
                    state_val = state->n_vector_output(flag, i);
                    for (const auto &i_node: local_nodes)
                        local_rhs[i_node] += state_val[i_comp] * get_shape_fns(qp).at(i_node) * JxW;
                }
                cell.M_inv.template vmult(local_projection, local_rhs);

                for (const auto &i_node: local_nodes)
                    projected_values[cell.local_dof_indices.at(cell.local_node_to_dof.at(i_node).at(i_comp))]
                            += local_projection[i_node] / connected_elements[cell.scalar_dof_indices.at(i_node)];
            }
        }
    }

    projected_values.compress(VectorOperation::add);

    Vector<double> out(projected_values);

    bool neighbour_is_not_of_correct_material = true;
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) == material_ids.end()) {

            for (unsigned int i_face = 0; i_face < cell->n_faces(); i_face++) {
                try {
                    neighbour_is_not_of_correct_material =
                            find(material_ids.begin(), material_ids.end(), cell->neighbor(i_face)->material_id()) ==
                            material_ids.end();
                } catch (...) {}
                if (not neighbour_is_not_of_correct_material)
                    break;
            }

            if (neighbour_is_not_of_correct_material) {
                cell->get_dof_indices(local_dof_indices);


                for (const auto &dof_index: local_dof_indices)
                    out[dof_index] = nan("");
            } else {
                neighbour_is_not_of_correct_material = true;
            }
        }
    }

    return out;
}

template<unsigned int dim>
Vector<double>
FEMesh<dim>::project_tensor_qp_field(const TensorOutputFlag &flag, const vector<unsigned int> &material_ids) const {
    Vector<double> projected_values;
    projected_values.reinit(tensor_dof_handler.n_dofs());

    vector<unsigned int> projection_dof_range(projection_dof_handler.n_dofs());
    iota(projection_dof_range.begin(), projection_dof_range.end(), 0);

    vector<vector<Vector<double>>> comp_wise_values(dim, vector<Vector<double >>(dim));
    for (const auto &i_comp: range)
        for (const auto &j_comp: range)
            comp_wise_values.at(i_comp).at(j_comp).reinit(projection_dof_handler.n_dofs());

    vector<types::global_dof_index> local_dof_indices(n_shape_fns);
    double JxW;
    Tensor<2, dim> state_val;
    StateBase<dim> *state;
    for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) != material_ids.end()) {

            cell->get_dof_indices(local_dof_indices);

            for (const auto &qp: qp_range) {
                JxW = get_JxW(cell->level(), cell->index(), qp);
                state = get_state(cell->level(), cell->index(), qp);
                state_val = state->tensor_output(flag);
                for (const auto &i: local_nodes)
                    for (const auto &i_comp: range)
                        for (const auto &j_comp: range)
                            comp_wise_values.at(i_comp).at(j_comp)[local_dof_indices[i]] +=
                                    state_val[i_comp][j_comp] * get_shape_fns(qp).at(i) * JxW;
            }
        }
    }

    for (const auto &i_comp: range)
        for (const auto &j_comp: range)
            projection_solver.solve(comp_wise_values.at(i_comp).at(j_comp));

    bool neighbour_is_not_of_correct_material = true;
    for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) == material_ids.end()) {

            for (unsigned int i_face = 0; i_face < cell->n_faces(); i_face++) {
                try {
                    neighbour_is_not_of_correct_material =
                            find(material_ids.begin(), material_ids.end(), cell->neighbor(i_face)->material_id()) ==
                            material_ids.end();
                } catch (...) {}
                if (not neighbour_is_not_of_correct_material)
                    break;
            }

            if (neighbour_is_not_of_correct_material) {
                cell->get_dof_indices(local_dof_indices);
                for (const auto &dof_index: local_dof_indices)
                    for (const auto &i_comp: range)
                        for (const auto &j_comp: range)
                            comp_wise_values.at(i_comp).at(j_comp)[dof_index] = nan("");
            } else {
                neighbour_is_not_of_correct_material = true;
            }
        }
    }

    for (const auto &i_comp: range)
        for (const auto &j_comp: range)
            for (const auto &k: projection_dof_range)
                projected_values[k * dim * dim + i_comp * dim + j_comp] = comp_wise_values.at(i_comp).at(j_comp)[k];

    return projected_values;
}

template<unsigned int dim>
Vector<double>
FEMesh<dim>::project_tensor_qp_field_el_wise(const TensorOutputFlag &flag,
                                             const vector<unsigned int> &material_ids) {
    if (n_connected_elements_material_combo.find(material_ids) == n_connected_elements_material_combo.end()) {
        add_material_combo_connected_elements(material_ids);
    }

    Vector<double> connected_elements(n_connected_elements_material_combo[material_ids]);

//    Vector<double> projected_values;
//    projected_values.reinit(tensor_dof_handler.n_dofs());

    PETScWrappers::MPI::Vector projected_values;
    projected_values.reinit(DoFTools::locally_owned_dofs_per_subdomain(tensor_dof_handler)[this_mpi_process],
                            mpi_communicator);

    Vector<double> local_rhs(n_shape_fns), local_projection(n_shape_fns);

    vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    vector<types::global_dof_index> tensor_local_dof_indices(n_shape_fns * dim * dim);
    double JxW;
    Tensor<2, dim> state_val;
    StateBase<dim> *state;

    for (const auto &cell: locally_relevant_elements) {
        if (find(material_ids.begin(), material_ids.end(), cell.cell_ptr->material_id()) != material_ids.end()) {
            for (const auto &i_comp: range)
                for (const auto &j_comp: range) {
                    local_rhs = 0.0;
                    local_projection = 0.0;
                    for (const auto &qp: qp_range) {
                        JxW = cell.JxWs.at(qp);
                        state = cell.states.at(qp);
                        state_val = state->tensor_output(flag);
                        for (const auto &i_node: local_nodes)
                            local_rhs[i_node] += state_val[i_comp][j_comp] * get_shape_fns(qp).at(i_node) * JxW;
                    }
                    cell.M_inv.template vmult(local_projection, local_rhs);

                    for (const auto &i_node: local_nodes)
                        projected_values[cell.tensor_dof_indices.at(i_node * dim * dim + i_comp * dim + j_comp)]
                                += local_projection[i_node] / connected_elements[cell.scalar_dof_indices.at(i_node)];
                }
        }
    }


    projected_values.compress(VectorOperation::add);

    Vector<double> out(projected_values);

    bool neighbour_is_not_of_correct_material = true;
    for (const auto &cell: tensor_dof_handler.active_cell_iterators()) {
        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) == material_ids.end()) {

            for (unsigned int i_face = 0; i_face < cell->n_faces(); i_face++) {
                try {
                    neighbour_is_not_of_correct_material =
                            find(material_ids.begin(), material_ids.end(), cell->neighbor(i_face)->material_id()) ==
                            material_ids.end();
                } catch (...) {}
                if (not neighbour_is_not_of_correct_material)
                    break;
            }

            if (neighbour_is_not_of_correct_material) {
                cell->get_dof_indices(tensor_local_dof_indices);


                for (const auto &dof_index: local_dof_indices)
                    out[dof_index] = nan("");
            } else {
                neighbour_is_not_of_correct_material = true;
            }
        }
    }


    return out;
}

template<unsigned int dim>
Vector<double>
FEMesh<dim>::project_n_tensor_qp_field(const nTensorOutputFlag &flag,
                                       const vector<unsigned int> &material_ids,
                                       const unsigned int &i) const {
    Vector<double> projected_values;
    projected_values.reinit(tensor_dof_handler.n_dofs());

    vector<unsigned int> projection_dof_range(projection_dof_handler.n_dofs());
    iota(projection_dof_range.begin(), projection_dof_range.end(), 0);

    vector<vector<Vector<double>>> comp_wise_values(dim, vector<Vector<double >>(dim));
    for (const auto &i_comp: range)
        for (const auto &j_comp: range)
            comp_wise_values.at(i_comp).at(j_comp).reinit(projection_dof_handler.n_dofs());

    vector<types::global_dof_index> local_dof_indices(n_shape_fns);
    double JxW;
    Tensor<2, dim> state_val;
    StateBase<dim> *state;
    for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) != material_ids.end()) {

            cell->get_dof_indices(local_dof_indices);

            for (const auto &qp: qp_range) {
                JxW = get_JxW(cell->level(), cell->index(), qp);
                state = get_state(cell->level(), cell->index(), qp);
                state_val = state->n_tensor_output(flag, i);
                for (const auto &i: local_nodes)
                    for (const auto &i_comp: range)
                        for (const auto &j_comp: range)
                            comp_wise_values.at(i_comp).at(j_comp)[local_dof_indices[i]] +=
                                    state_val[i_comp][j_comp] * get_shape_fns(qp).at(i) * JxW;
            }
        }
    }

    for (const auto &i_comp: range)
        for (const auto &j_comp: range)
            projection_solver.solve(comp_wise_values.at(i_comp).at(j_comp));

    bool neighbour_is_not_of_correct_material = true;
    for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) == material_ids.end()) {

            for (unsigned int i_face = 0; i_face < cell->n_faces(); i_face++) {
                try {
                    neighbour_is_not_of_correct_material =
                            find(material_ids.begin(), material_ids.end(), cell->neighbor(i_face)->material_id()) ==
                            material_ids.end();
                } catch (...) {}
                if (not neighbour_is_not_of_correct_material)
                    break;
            }

            if (neighbour_is_not_of_correct_material) {
                cell->get_dof_indices(local_dof_indices);
                for (const auto &dof_index: local_dof_indices)
                    for (const auto &i_comp: range)
                        for (const auto &j_comp: range)
                            comp_wise_values.at(i_comp).at(j_comp)[dof_index] = nan("");
            } else {
                neighbour_is_not_of_correct_material = true;
            }
        }
    }

    for (const auto &i_comp: range)
        for (const auto &j_comp: range)
            for (const auto &k: projection_dof_range)
                projected_values[k * dim * dim + i_comp * dim + j_comp] = comp_wise_values.at(i_comp).at(j_comp)[k];

    return projected_values;
}

template<unsigned int dim>
Vector<double>
FEMesh<dim>::project_n_tensor_qp_field_el_wise(const nTensorOutputFlag &flag,
                                               const vector<unsigned int> &material_ids,
                                               const unsigned int &i) {
    if (n_connected_elements_material_combo.find(material_ids) == n_connected_elements_material_combo.end()) {
        add_material_combo_connected_elements(material_ids);
    }

    Vector<double> connected_elements(n_connected_elements_material_combo[material_ids]);

//    Vector<double> projected_values;
//    projected_values.reinit(tensor_dof_handler.n_dofs());

    PETScWrappers::MPI::Vector projected_values;
    projected_values.reinit(DoFTools::locally_owned_dofs_per_subdomain(tensor_dof_handler)[this_mpi_process],
                            mpi_communicator);

    Vector<double> local_rhs(n_shape_fns), local_projection(n_shape_fns);

    vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    vector<types::global_dof_index> tensor_local_dof_indices(n_shape_fns * dim * dim);
    double JxW;
    Tensor<2, dim> state_val;
    StateBase<dim> *state;

    for (const auto &cell: locally_relevant_elements) {
        if (find(material_ids.begin(), material_ids.end(), cell.cell_ptr->material_id()) != material_ids.end()) {
            for (const auto &i_comp: range)
                for (const auto &j_comp: range) {
                    local_rhs = 0.0;
                    local_projection = 0.0;
                    for (const auto &qp: qp_range) {
                        JxW = cell.JxWs.at(qp);
                        state = cell.states.at(qp);
                        state_val = state->n_tensor_output(flag, i);
                        for (const auto &i_node: local_nodes)
                            local_rhs[i_node] += state_val[i_comp][j_comp] * get_shape_fns(qp).at(i_node) * JxW;
                    }
                    cell.M_inv.template vmult(local_projection, local_rhs);

                    for (const auto &i_node: local_nodes)
                        projected_values[cell.tensor_dof_indices.at(i_node * dim * dim + i_comp * dim + j_comp)]
                                += local_projection[i_node] / connected_elements[cell.scalar_dof_indices.at(i_node)];
                }
        }
    }


    projected_values.compress(VectorOperation::add);

    Vector<double> out(projected_values);

    bool neighbour_is_not_of_correct_material = true;
    for (const auto &cell: tensor_dof_handler.active_cell_iterators()) {
        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) == material_ids.end()) {

            for (unsigned int i_face = 0; i_face < cell->n_faces(); i_face++) {
                try {
                    neighbour_is_not_of_correct_material =
                            find(material_ids.begin(), material_ids.end(), cell->neighbor(i_face)->material_id()) ==
                            material_ids.end();
                } catch (...) {}
                if (not neighbour_is_not_of_correct_material)
                    break;
            }

            if (neighbour_is_not_of_correct_material) {
                cell->get_dof_indices(tensor_local_dof_indices);


                for (const auto &dof_index: local_dof_indices)
                    out[dof_index] = nan("");
            } else {
                neighbour_is_not_of_correct_material = true;
            }
        }
    }


    return out;
}

template<unsigned int dim>
void FEMesh<dim>::update_states() {
    for (const auto &state_item: states)
        state_item.second->update();

    for (auto &el: locally_relevant_elements)
        for(auto &state: el.states)
            state->update();
}


template<unsigned int dim>
Vector<double>
FEMesh<dim>::mesh_output_values(const MeshOutputFlag &flag) const {
    switch (flag) {
        case MeshOutputFlag::MATERIAL_ID:
            return material_id_output();
        case MeshOutputFlag::BOUNDARY_ID:
            return boundary_id_output();
        default:
            throw NotImplemented("mesh_output_values is not implemented for flag '" + to_string(flag) + "'.");
    }
}

template<unsigned int dim>
Vector<double> FEMesh<dim>::material_id_output() const {
    Vector<double> material_ids(triangulation.n_active_cells());
    unsigned int count = 0;

    for (const auto &cell: dof_handler.active_cell_iterators()) {
        material_ids[count] = cell->material_id();
        count++;
    }

    return material_ids;
}

template<unsigned int dim>
Vector<double> FEMesh<dim>::boundary_id_output() const {
    Vector<double> boundary_ids(triangulation.n_active_cells());
    boundary_ids = 0;
    unsigned int count = 0;
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        if (cell->at_boundary()) {
            for (const auto &face: cell->face_iterators()) {
                if (face->at_boundary()) {
                    boundary_ids[count] = face->boundary_id();
                    break;
                }
            }
        } else {
            boundary_ids[count] = nan("");
        }
        count++;
    }

    return boundary_ids;
}

template<unsigned int dim>
void FEMesh<dim>::output_scalar_averages(const Time *time,
                                         const map<ScalarOutputFlag, vector<unsigned int>> &scalar_volume_average) const {
    double value, vol;
    double JxW, state_val;
    string out_name;
    string file_contents;
    StateBase<dim> *state;
    vector<unsigned int> material_ids;
    for (const auto &out_f: scalar_volume_average) {
        value = 0;
        vol = 0;
        material_ids = out_f.second;

        for (const auto &cell: locally_relevant_elements) {
            if (find(material_ids.begin(), material_ids.end(), cell.cell_ptr->material_id()) != material_ids.end()) {
                for (const auto &qp: qp_range) {
                    JxW = cell.JxWs.at(qp);
                    state = cell.states.at(qp);
                    state_val = state->scalar_output(out_f.first);
                    value += state_val * JxW;
                    vol += JxW;
                }
            }
        }
        value = Utilities::MPI::sum(value, mpi_communicator);
        vol = Utilities::MPI::sum(vol, mpi_communicator);

        if (this_mpi_process == 0) {
            value /= vol;
            out_name = OutputFlags::to_string(out_f.first) + ".csv";

            if (time->get_stage() == 0 && time->get_timestep() == 0) {
                file_contents = "stage, timestep, time, " + to_string(out_f.first) + "\n";
            } else {
                ifstream input((output_dir + out_name).c_str());
                stringstream buffer;
                buffer << input.rdbuf();
                file_contents = buffer.str();
                input.close();
            }

            ofstream out_file((output_dir + out_name).c_str());
            out_file << file_contents;
            out_file << time->get_stage() << ", "
                     << time->get_timestep() << ", "
                     << time->current() << ", "
                     << value << endl;
            out_file.close();

            file_contents = "";
        }

    }
}


template<unsigned int dim>
void FEMesh<dim>::output_vector_averages(const Time *time,
                                         const map<VectorOutputFlag, vector<unsigned int>> &vector_volume_average) const {
    double vol, JxW;
    Tensor<1, dim> value, state_val;

    string out_name;
    string file_contents;
    StateBase<dim> *state;
    vector<unsigned int> material_ids;
    for (const auto &out_f: vector_volume_average) {
        value = 0;
        vol = 0;
        material_ids = out_f.second;

        for (const auto &cell: locally_relevant_elements) {
            if (find(material_ids.begin(), material_ids.end(), cell.cell_ptr->material_id()) != material_ids.end()) {
                for (const auto &qp: qp_range) {
                    JxW = cell.JxWs.at(qp);
                    state = cell.states.at(qp);
                    state_val = state->vector_output(out_f.first);
                    value += state_val * JxW;
                    vol += JxW;
                }
            }
        }
        value = Utilities::MPI::sum(value, mpi_communicator);
        vol = Utilities::MPI::sum(vol, mpi_communicator);

        if (this_mpi_process == 0) {
            value /= vol;
            out_name = "VolumeAveraged" + OutputFlags::to_string<dim>(out_f.first).at(0) + ".csv";

            if (time->get_stage() == 0 && time->get_timestep() == 0) {
//            file_contents = "stage, timestep, time" + to_string(out_f.first) + "\n";
                file_contents = "stage, timestep, time";

                for (const auto &comp: OutputFlags::to_string<dim>(out_f.first))
                    file_contents += ", " + comp;

                file_contents += "\n";

            } else {
                ifstream input((output_dir + out_name).c_str());
                stringstream buffer;
                buffer << input.rdbuf();
                file_contents = buffer.str();
                input.close();
            }

            ofstream out_file((output_dir + out_name).c_str());
            out_file << file_contents;
            out_file << time->get_stage() << ", "
                     << time->get_timestep() << ", "
                     << time->current();
            for (const auto &comp: range)
                out_file << ", " << value[comp];

            out_file << endl;
            out_file.close();

            file_contents = "";
        }
    }

}


template<unsigned int dim>
void FEMesh<dim>::output_tensor_averages(const Time *time,
                                         const map<TensorOutputFlag, vector<unsigned int>> &tensor_volume_average) const {
    double vol, JxW;
    Tensor<2, dim> value, state_val;

    string out_name;
    string file_contents;
    StateBase<dim> *state;
    vector<unsigned int> material_ids;
    for (const auto &out_f: tensor_volume_average) {
        value = 0;
        vol = 0;
        material_ids = out_f.second;

        for (const auto &cell: locally_relevant_elements) {
            if (find(material_ids.begin(), material_ids.end(), cell.cell_ptr->material_id()) != material_ids.end()) {
                for (const auto &qp: qp_range) {
                    JxW = cell.JxWs.at(qp);
                    state = cell.states.at(qp);
                    state_val = state->tensor_output(out_f.first);
                    value += state_val * JxW;
                    vol += JxW;
                }
            }
        }
        value = Utilities::MPI::sum(value, mpi_communicator);
        vol =  Utilities::MPI::sum(vol, mpi_communicator);

        if (this_mpi_process == 0) {
            value /= vol;
            out_name = "VolumeAveraged" + OutputFlags::to_string<dim>(out_f.first).at(0);
            out_name = out_name.substr(0, out_name.size() - 2);
            out_name += ".csv";

            if (time->get_stage() == 0 && time->get_timestep() == 0) {
                file_contents = "stage, timestep, time";
                for (const auto &comp: OutputFlags::to_string<dim>(out_f.first))
                    file_contents += ", " + comp;

                file_contents += "\n";

            } else {
                ifstream input((output_dir + out_name).c_str());
                stringstream buffer;
                buffer << input.rdbuf();
                file_contents = buffer.str();
                input.close();
            }

            ofstream out_file((output_dir + out_name).c_str());
            out_file << file_contents;
            out_file << time->get_stage() << ", "
                     << time->get_timestep() << ", "
                     << time->current();

            for (const auto &icomp: range)
                for (const auto &jcomp: range)
                    out_file << ", " << value[icomp][jcomp];

            out_file << endl;
            out_file.close();

            file_contents = "";
        }
    }
}

template<unsigned int dim>
void FEMesh<dim>::add_material_combo_connected_elements(const material_id_combination &mat_combo) {
    n_connected_elements_material_combo.insert({mat_combo, PETScWrappers::MPI::Vector()});
    n_connected_elements_material_combo.at(mat_combo).reinit(
            DoFTools::locally_owned_dofs_per_subdomain(scalar_dof_handler)[this_mpi_process],
            mpi_communicator);
    vector<types::global_dof_index> scalar_dof_indices(fe.base_element(0).n_dofs_per_cell());
    vector<PetscScalar> ones(scalar_dof_indices.size(), 1);
    for (const auto &cell: scalar_dof_handler.active_cell_iterators()) {
        if (cell->subdomain_id() == this_mpi_process) {
            if (find(mat_combo.begin(), mat_combo.end(), cell->material_id()) != mat_combo.end()) {
                cell->get_dof_indices(scalar_dof_indices);
                n_connected_elements_material_combo.at(mat_combo).add(scalar_dof_indices, ones);
            }
        }
    }
    n_connected_elements_material_combo.at(mat_combo).compress(VectorOperation::add);
}


#endif //SOLVER_FEMESH_H
