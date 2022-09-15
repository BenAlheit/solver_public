#ifndef SOLVER_FEMESH_H
#define SOLVER_FEMESH_H

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/base/mpi.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

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

template<unsigned int dim>
class FEMesh {
public:
//    FEMesh(unsigned int order,
//           map<unsigned int, Material<dim> *> materials);
    FEMesh(unsigned int order,
           const map<unsigned int, Material<dim> *> &materials,
           const Triangulation<dim> &in_triangulation,
           const unsigned int quadrature_dif = 1);

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

    const Triangulation<dim> &get_triangulation() const { return triangulation; };

    const FESystem<dim> &get_fe() const { return fe; };

    const DoFHandler<dim> &get_dof_handler() const { return dof_handler; };

    const DoFHandler<dim> &get_scalar_dof_handler() const { return projection_dof_handler; };

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
    project_vector_qp_field(const VectorOutputFlag &flag, const vector<unsigned int> &material_ids) const;

    Vector<double>
    project_tensor_qp_field(const TensorOutputFlag &flag, const vector<unsigned int> &material_ids) const;

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

    DoFHandler<dim> tensor_dof_handler;

    vector<array<unsigned int, dim>> local_node_to_dof;
    map<unsigned int, Material<dim> *> materials;

    vector<vector<double>> shape_fns; // @ qp -> node array
    unordered_map<array<unsigned int, 3>, vector<Tensor<1, dim>>, array_hash> grad_shape_fns; // @ level, el_id, qp -> node array
    unordered_map<array<unsigned int, 3>, double, array_hash> JxWs; // @ level, el_id, qp -> node array
    unordered_map<array<unsigned int, 3>, StateBase<dim> *, array_hash> states; // @ level, el_id, qp -> material state

    unsigned int dofs_per_cell;
    unsigned int n_q_points;
    unsigned int n_shape_fns;

    Vector<double>
    material_id_output() const;

    Vector<double>
    boundary_id_output() const;
};

template<unsigned int dim>
FEMesh<dim>::FEMesh(unsigned int order,
                    const map<unsigned int, Material<dim> *> &materials,
                    const Triangulation<dim> &in_triangulation,
                    const unsigned int quadrature_dif)
        : mpi_communicator(MPI_COMM_WORLD), n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
          this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)), pcout(cout, (this_mpi_process == 0)),
          fe(FE_Q<dim>(order), dim)
//        , triangulation(in_triangulation)
        , dof_handler(triangulation), quadrature_formula(fe.degree + quadrature_dif), materials(materials),
          projection_dof_handler(triangulation), tensor_dof_handler(triangulation) {
    pcout << "n mpi processes: " << n_mpi_processes << endl;

    triangulation.copy_triangulation(in_triangulation);
    dof_handler.reinit(triangulation);
    iota(range.begin(), range.end(), 0);

    GridTools::partition_triangulation(n_mpi_processes, triangulation);
    dof_handler.distribute_dofs(fe);
    DoFRenumbering::subdomain_wise(dof_handler);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    dofs_per_cell = fe.n_dofs_per_cell();
    n_q_points = quadrature_formula.size();
    n_shape_fns = fe_values.get_fe().base_element(0).n_dofs_per_cell();

    local_nodes = vector<unsigned int>(n_shape_fns);
    iota(local_nodes.begin(), local_nodes.end(), 0);

    local_node_to_dof = vector<array<unsigned int, dim>>(n_shape_fns);

    for (auto const &i_node: local_nodes)
        for (auto const &i_comp: range)
            local_node_to_dof.at(i_node).at(i_comp) = fe_values.get_fe().component_to_system_index(i_comp, i_node);

    vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    dof_range = vector<unsigned int>(dofs_per_cell);
    iota(dof_range.begin(), dof_range.end(), 0);


    vector<Tensor<1, dim>> el_grad_shape_fns(n_shape_fns);
    shape_fns = vector<vector<double>>(n_q_points, vector<double>(n_shape_fns));
    qp_range = vector<unsigned int>(n_q_points);
    iota(qp_range.begin(), qp_range.end(), 0);

    for (const auto &cell: dof_handler.active_cell_iterators())
        if (cell->subdomain_id() == this_mpi_process) {
            fe_values.reinit(cell);
            for (const unsigned int q_point: qp_range) {
                for (const auto &i_node: local_nodes) {
                    shape_fns.at(q_point).at(i_node) = fe_values.shape_value(local_node_to_dof.at(i_node).at(0),
                                                                             q_point);
                }
            }
            break;
        }

    for (const auto &cell: dof_handler.active_cell_iterators())
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

    projection_dof_handler.initialize(triangulation, fe.base_element(0));
    projection_dof_handler.distribute_dofs(fe.base_element(0));

    DynamicSparsityPattern dsp(projection_dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(projection_dof_handler, dsp);
    SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dsp);
    projection_matrix.reinit(sparsity_pattern);
    vector<types::global_dof_index> projection_local_dof_indices(n_shape_fns);
    double JxW;

    for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
//        fe_values.reinit(cell);
        cell->get_dof_indices(projection_local_dof_indices);

        for (const auto &qp: qp_range) {
            JxW = get_JxW(cell->level(), cell->index(), qp);
            for (const auto &i: local_nodes)
                for (const auto &j: local_nodes)
                    projection_matrix.add(projection_local_dof_indices[i],
                                          projection_local_dof_indices[j],
                                          shape_fns.at(qp).at(i) * shape_fns.at(qp).at(j) * JxW);
        }
    }

    projection_solver.template initialize(projection_matrix);

//    DoFHandler<dim> tensor_dof_handler(triangulation);
    FESystem<dim, dim> tensor_fe(FE_Q<dim, dim>(fe.degree), dim * dim);
//    tensor_dof_handler.distribute_dofs(tensor_fe);
    tensor_dof_handler.initialize(triangulation, tensor_fe);
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

            for(unsigned int i_face=0; i_face < cell->n_faces(); i_face++){
                try {
                    neighbour_is_not_of_correct_material = find(material_ids.begin(), material_ids.end(), cell->neighbor(i_face)->material_id()) == material_ids.end();
                } catch (...) { }
                if(not neighbour_is_not_of_correct_material)
                    break;
            }

            if(neighbour_is_not_of_correct_material) {
                cell->get_dof_indices(local_dof_indices);


                for (const auto &dof_index: local_dof_indices)
                    projected_values[dof_index] = nan("");
            }else{
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

    for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
        if (find(material_ids.begin(), material_ids.end(), cell->material_id()) == material_ids.end()) {

            cell->get_dof_indices(local_dof_indices);
            for (const auto &dof_index: local_dof_indices)
                for (const auto &i_comp: range)
                    comp_wise_values.at(i_comp)[dof_index] = nan("");
        }
    }

    for (const auto &i_comp: range)
        for (const auto &k: projection_dof_range)
            projected_values[k * dim + i_comp] = comp_wise_values.at(i_comp)[k];

    return projected_values;
}


template<unsigned int dim>
Vector<double>
FEMesh<dim>::project_tensor_qp_field(const TensorOutputFlag &flag, const vector<unsigned int> &material_ids) const {
    Vector<double> projected_values;
    projected_values.reinit(tensor_dof_handler.n_dofs());

    vector<unsigned int> projection_dof_range(projection_dof_handler.n_dofs());
    iota(projection_dof_range.begin(), projection_dof_range.end(), 0);

    vector<vector<Vector<double>>> comp_wise_values(dim, vector<Vector<double>>(dim));
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

            for(unsigned int i_face=0; i_face < cell->n_faces(); i_face++){
                try {
                    neighbour_is_not_of_correct_material = find(material_ids.begin(), material_ids.end(), cell->neighbor(i_face)->material_id()) == material_ids.end();
                } catch (...) { }
                if(not neighbour_is_not_of_correct_material)
                    break;
            }

            if(neighbour_is_not_of_correct_material) {
                cell->get_dof_indices(local_dof_indices);
                for (const auto &dof_index: local_dof_indices)
                    for (const auto &i_comp: range)
                        for (const auto &j_comp: range)
                            comp_wise_values.at(i_comp).at(j_comp)[dof_index] = nan("");
            }else{
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
void FEMesh<dim>::update_states() {
    for (const auto &state_item: states)
        state_item.second->update();
}


template<unsigned int dim>
Vector<double>
FEMesh<dim>::mesh_output_values(const MeshOutputFlag &flag) const {
    switch (flag) {
        case MATERIAL_ID:
            return material_id_output();
        case BOUNDARY_ID:
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

        for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
            if (find(material_ids.begin(), material_ids.end(), cell->material_id()) != material_ids.end()) {
                for (const auto &qp: qp_range) {
                    JxW = get_JxW(cell->level(), cell->index(), qp);
                    state = get_state(cell->level(), cell->index(), qp);
                    state_val = state->scalar_output(out_f.first);
                    value += state_val * JxW;
                    vol += JxW;
                }
            }
        }

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

        for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
            if (find(material_ids.begin(), material_ids.end(), cell->material_id()) != material_ids.end()) {
                for (const auto &qp: qp_range) {
                    JxW = get_JxW(cell->level(), cell->index(), qp);
                    state = get_state(cell->level(), cell->index(), qp);
                    state_val = state->vector_output(out_f.first);
                    value += state_val * JxW;
                    vol += JxW;
                }
            }
        }

        value /= vol;
        out_name = "VolumeAveraged" + OutputFlags::to_string<dim>(out_f.first).at(0) + ".csv";

        if (time->get_stage() == 0 && time->get_timestep() == 0) {
            file_contents = "stage, timestep, time" + to_string(out_f.first) + "\n";

            for(const auto & comp : OutputFlags::to_string<dim>(out_f.first))
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
        for(const auto & comp : range)
            out_file << ", " << value[comp];

        out_file << endl;
        out_file.close();

        file_contents = "";
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

        for (const auto &cell: projection_dof_handler.active_cell_iterators()) {
            if (find(material_ids.begin(), material_ids.end(), cell->material_id()) != material_ids.end()) {
                for (const auto &qp: qp_range) {
                    JxW = get_JxW(cell->level(), cell->index(), qp);
                    state = get_state(cell->level(), cell->index(), qp);
                    state_val = state->tensor_output(out_f.first);
                    value += state_val * JxW;
                    vol += JxW;
                }
            }
        }

        value /= vol;
        out_name = "VolumeAveraged" + OutputFlags::to_string<dim>(out_f.first).at(0);
        out_name = out_name.substr(0, out_name.size()-2);
        out_name += ".csv";

        if (time->get_stage() == 0 && time->get_timestep() == 0) {
            file_contents = "stage, timestep, time";
            for(const auto & comp : OutputFlags::to_string<dim>(out_f.first))
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

        for(const auto & icomp : range)
            for(const auto & jcomp : range)
                out_file << ", " << value[icomp][jcomp];

        out_file << endl;
        out_file.close();

        file_contents = "";
    }
}


#endif //SOLVER_FEMESH_H
