//
// Created by alhei on 2022/08/29.
//

#ifndef SOLVER_SOLVER_H
#define SOLVER_SOLVER_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <mpi.h>
#include <fstream>

#include "problem/Problem.h"
#include "materials/MaterialState.h"
#include "materials/OutputFlags.h"


namespace Solver {
    using namespace dealii;
//    using namespace dealii::LinearAlgebraPETSc;
    using namespace std;

    template<unsigned int dim>
    class Solver {
    public:
        explicit Solver(Problem<dim> *problem)
                : problem(problem), dsp(problem->get_mesh()->get_dof_handler().n_dofs(),
                                        problem->get_mesh()->get_dof_handler().n_dofs()) {
            iota(range.begin(), range.end(), 0);

            output_dir_name = "./" + problem->get_name() + "/";
            problem->get_mesh()->set_output_dir(output_dir_name + "VolumeAverages/");

            locally_owned_dofs_per_proc =
                    DoFTools::locally_owned_dofs_per_subdomain(problem->get_mesh()->get_dof_handler());
            locally_relevant_dofs_per_proc =
                    DoFTools::locally_relevant_dofs_per_subdomain(problem->get_mesh()->get_dof_handler());

            locally_owned_dofs = locally_owned_dofs_per_proc[problem->get_mesh()->get_this_mpi_processes()];
            locally_relevant_dofs = locally_relevant_dofs_per_proc[problem->get_mesh()->get_this_mpi_processes()];
            cout << "Process " << problem->get_mesh()->get_this_mpi_processes() << " locally owned dofs: " << endl;
            locally_owned_dofs.print(cout);
            cout << "Process " << problem->get_mesh()->get_this_mpi_processes() << " locally relevant dofs: " << endl;
            locally_relevant_dofs.print(cout);

            u.reinit(locally_owned_dofs,
                     problem->get_mesh()->get_mpi_communicator());
            du.reinit(locally_owned_dofs,
                      problem->get_mesh()->get_mpi_communicator());

            locally_relevant_u.reinit(locally_owned_dofs,
                                      locally_relevant_dofs,
                                      problem->get_mesh()->get_mpi_communicator());

            system_rhs.reinit(locally_owned_dofs,
                              problem->get_mesh()->get_mpi_communicator());
            u = 0;
            du = 0;
        };

        void solve();

    protected:

    private:


        bool init_step(bool init_solve = false);


        void assemble_system(bool init = false, bool init_solve = false);

//        void assemble_system_old(bool init = false, bool init_solve = false);

        void do_time_step(bool init_solve = false);

        void nr();


        void output_results() const;

        string output_dir_name;
        Problem<dim> *problem;

        array<unsigned int, dim> range;

        vector<IndexSet> locally_owned_dofs_per_proc;
        vector<IndexSet> locally_relevant_dofs_per_proc;
        IndexSet locally_owned_dofs;
        IndexSet locally_relevant_dofs;

        DynamicSparsityPattern dsp;

        AffineConstraints<double> du_constraints;
        AffineConstraints<double> dummy_constraints;

        PETScWrappers::MPI::SparseMatrix system_matrix;

        PETScWrappers::MPI::Vector du;
        PETScWrappers::MPI::Vector u;
        PETScWrappers::MPI::Vector locally_relevant_u;
        PETScWrappers::MPI::Vector system_rhs;

    };

    template<unsigned int dim>
    bool Solver<dim>::init_step(bool init_solve) {

        bool output = problem->step(du_constraints);
        du = 0;

        DoFTools::make_sparsity_pattern(problem->get_mesh()->get_dof_handler(),
                                        dsp,
                                        du_constraints,
                                        false);

        system_matrix.reinit(locally_owned_dofs,
                             locally_owned_dofs,
                             dsp,
                             problem->get_mesh()->get_mpi_communicator());

        assemble_system(true, init_solve);
        double solver_res = 1e-8 * system_rhs.l2_norm();

        SolverControl solver_control(5000, solver_res);

        PETScWrappers::SolverCG cg(solver_control,
                                   this->problem->get_mesh()->get_mpi_communicator());
//        PETScWrappers::SolverGMRES cg(solver_control,
//                                      this->problem->get_mesh()->get_mpi_communicator());

        PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);

//        LinearAlgebraPETSc::MPI::PreconditionAMG preconditioner;
//        LinearAlgebraPETSc::MPI::PreconditionAMG::AdditionalData data;
//        preconditioner.initialize(system_matrix, data);

        cg.solve(system_matrix, du, system_rhs, preconditioner);
        du_constraints.template distribute(du);
        u += du;

        dummy_constraints.clear();
        dummy_constraints.template copy_from(du_constraints);
        du_constraints.clear();
        du_constraints.template copy_from(dummy_constraints);
        for (const auto &line: du_constraints.get_lines()) {
            du_constraints.set_inhomogeneity(line.index, 0);
        }
        du_constraints.close();
        du_constraints.template distribute(du);

        return output;
    }


    template<unsigned int dim>
    void Solver<dim>::assemble_system(bool init_step, bool init_solve) {
        system_matrix = 0.0;
        system_rhs = 0.0;

        u.compress(VectorOperation::unknown);
        locally_relevant_u = u;

        FullMatrix<double> cell_matrix(problem->get_mesh()->get_dofs_per_cell(),
                                       problem->get_mesh()->get_dofs_per_cell());
        Vector<double> cell_rhs(problem->get_mesh()->get_dofs_per_cell());


        StateBase<dim> *qp_state;
        Material<dim> *material;
        array<unsigned int, dim> i_node_local_dofs, j_node_local_dofs;

        vector<Tensor<1, dim>> spatial_grad_phis(problem->get_mesh()->get_n_shape_fns());
        vector<Tensor<1, dim>> ref_grad_phis(problem->get_mesh()->get_n_shape_fns());

        Tensor<1, dim> grad_phi_i, grad_phi_j, r_i;
        Tensor<2, dim> Grad_u, F, F_inv, tau, k_ij;
        SymmetricTensor<2, dim> sym_tau;
        SymmetricTensor<4, dim> sym_tangent;
        Tensor<3, dim> grad_phi_i_tangent;
        Tensor<1, dim> grad_phi_i_tau;
        SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;
        double JxW;

        Vector<double> local_u;
        local_u.reinit(problem->get_mesh()->get_dofs_per_cell());
        vector<PetscScalar> std_local_u(problem->get_mesh()->get_dofs_per_cell());

        for (const auto &cell: problem->get_mesh()->get_locally_relevant_elements()) {
            cell_matrix = 0;
            cell_rhs = 0;

            locally_relevant_u.extract_subvector_to(cell.local_dof_indices, std_local_u);
            for (const auto &i_dof: problem->get_mesh()->get_dof_range())
                local_u[i_dof] = (double) std_local_u.at(i_dof);

            for (const auto &qp: problem->get_mesh()->get_qp_range()) {
                JxW = cell.JxWs.at(qp);
                Grad_u = cell.grad_field(qp, local_u);
                F = I + Grad_u;
                F_inv = invert(F);
                qp_state = cell.states.at(qp);

                qp_state->F_n1 = F;

                if (not init_step or init_solve) {
                    material = problem->get_mesh()->get_material(cell.cell_ptr->material_id());
                    material->set_state(qp_state);
                    material->update_stress_and_tangent(problem->get_time()->get_current_delta_t());
                }

                sym_tau = symmetrize(qp_state->tau_n1);

                ref_grad_phis = cell.grad_shape_fns.at(qp);
                for (const auto &i_node: problem->get_mesh()->get_local_nodes())
                    spatial_grad_phis.at(i_node) = ref_grad_phis.at(i_node) * F_inv;

                for (const auto &i_node: problem->get_mesh()->get_local_nodes()) {
                    i_node_local_dofs = problem->get_mesh()->get_local_node_to_dofs(i_node);
                    grad_phi_i = spatial_grad_phis.at(i_node);

                    grad_phi_i_tau = sym_tau * grad_phi_i;
                    grad_phi_i_tangent = grad_phi_i * qp_state->c_n1;

                    r_i = -grad_phi_i_tau * JxW;
                    for (const auto &i: range)
                        cell_rhs[i_node_local_dofs.at(i)] += r_i[i];

                    for (const auto &j_node: problem->get_mesh()->get_local_nodes()) {
                        j_node_local_dofs = problem->get_mesh()->get_local_node_to_dofs(j_node);
                        grad_phi_j = spatial_grad_phis.at(j_node);
                        k_ij = 0;
                        k_ij = (grad_phi_i_tangent * grad_phi_j + I * (grad_phi_i_tau * grad_phi_j)) * JxW;

                        for (const auto &i: range)
                            for (const auto &j: range)
                                cell_matrix[i_node_local_dofs.at(i)][j_node_local_dofs.at(j)] += k_ij[i][j];
                    }
                }
            }

            du_constraints.distribute_local_to_global(cell_matrix,
                                                      cell_rhs,
                                                      cell.local_dof_indices,
                                                      system_matrix,
                                                      system_rhs);
        }

        system_matrix.compress(VectorOperation::add);
        system_rhs.compress(VectorOperation::add);

    }

    template<unsigned int dim>
    void Solver<dim>::do_time_step(bool init_solve) {
        problem->get_mesh()->get_pcout() << "***********************************" << endl;
        problem->get_mesh()->get_pcout() << "Stage: " << problem->get_time()->get_stage() << endl;
        problem->get_mesh()->get_pcout() << "Time step: " << problem->get_time()->get_timestep() << endl;
        problem->get_mesh()->get_pcout() << "Time: " << problem->get_time()->current() << " s" << endl;
        bool output = init_step(init_solve);
        nr();
        if (output) {
            output_results();
        }
        problem->get_mesh()->update_states();
        problem->get_mesh()->output_averages(problem->get_time(),
                                             problem->get_stage().get_scalar_volume_average_outputs(),
                                             problem->get_stage().get_vector_volume_average_outputs(),
                                             problem->get_stage().get_tensor_volume_average_outputs());
    }

    template<unsigned int dim>
    void Solver<dim>::nr() {
//        SolverControl solver_control(u.size(), 1e-8 * system_rhs.l2_norm());
////        PETScWrappers::SolverCG cg(solver_control,
////                                   this->problem->get_mesh()->get_mpi_communicator());
//        PETScWrappers::SolverGMRES cg(solver_control,
//                                      this->problem->get_mesh()->get_mpi_communicator());
//
//        PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
        problem->get_mesh()->get_pcout() << "Initial step residual: " << system_rhs.l2_norm() << endl;

        for (unsigned int it = 0; it < 7; it++) {
            assemble_system();
            problem->get_mesh()->get_pcout() << "Iteration: " << it << "\t residual norm: " << system_rhs.l2_norm()
                                             << endl;
            SolverControl solver_control(u.size(), 1e-8 * system_rhs.l2_norm());
        PETScWrappers::SolverCG cg(solver_control,
                                   this->problem->get_mesh()->get_mpi_communicator());
//            PETScWrappers::SolverGMRES cg(solver_control,
//                                          this->problem->get_mesh()->get_mpi_communicator());
            PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);

//            LinearAlgebraPETSc::MPI::PreconditionAMG preconditioner;
//            LinearAlgebraPETSc::MPI::PreconditionAMG::AdditionalData data;
//            preconditioner.initialize(system_matrix, data);

            cg.solve(system_matrix, du, system_rhs, preconditioner);

            du_constraints.template distribute(du);

            u += du;
        }
    }

    template<unsigned int dim>
    void Solver<dim>::solve() {
        output_results();
        problem->get_mesh()->output_averages(problem->get_time(),
                                             problem->get_stage().get_scalar_volume_average_outputs(),
                                             problem->get_stage().get_vector_volume_average_outputs(),
                                             problem->get_stage().get_tensor_volume_average_outputs());
        do_time_step(true);
        bool more_stages = true;
        while (more_stages) {
            while (problem->get_time()->current() < problem->get_time()->end())
                do_time_step();

            more_stages = not problem->last_stage();
            if (more_stages)
                problem->next_stage();
        }
    }

    template<unsigned int dim>
    void Solver<dim>::output_results() const {
        const Vector<double> localized_solution(u);

        vector<Vector<double>> scalar_values;
        vector<string> scalar_names;

        for (const auto &out_item: problem->get_stage().get_scalar_outputs()) {
            scalar_values.push_back(
                    problem->get_mesh()->project_scalar_qp_field_el_wise(out_item.first, out_item.second));
            scalar_names.push_back(OutputFlags::to_string(out_item.first));
        }

        for (const auto &out_item: problem->get_stage().get_n_scalar_outputs()) {
            for (unsigned int i = 0; i < out_item.first.second; i++) {
                scalar_values.push_back(problem->get_mesh()->project_n_scalar_qp_field_el_wise(out_item.first.first,
                                                                                               out_item.second,
                                                                                               i));
                scalar_names.push_back(OutputFlags::to_string(out_item.first.first, i));
            }
        }

        vector<Vector<double>> vector_values;
        vector<vector<string>> vector_names;

        for (const auto &out_item: problem->get_stage().get_vector_outputs()) {
            vector_values.push_back(
                    problem->get_mesh()->project_vector_qp_field_el_wise(out_item.first, out_item.second));
            auto names = OutputFlags::to_string<dim>(out_item.first);
            vector_names.push_back(names);
        }

        for (const auto &out_item: problem->get_stage().get_n_vector_outputs()) {
            for (unsigned int i = 0; i < out_item.first.second; i++) {
                vector_values.push_back(problem->get_mesh()->project_n_vector_qp_field_el_wise(out_item.first.first,
                                                                                               out_item.second,
                                                                                               i));
                auto names = OutputFlags::to_string<dim>(out_item.first.first, i);
                vector_names.push_back(names);
            }
        }

        vector<Vector<double>> tensor_values;
        vector<vector<string>> tensor_names;

        for (const auto &out_item: problem->get_stage().get_tensor_outputs()) {
            tensor_values.push_back(
                    problem->get_mesh()->project_tensor_qp_field_el_wise(out_item.first, out_item.second));
            auto names = OutputFlags::to_string<dim>(out_item.first);
            tensor_names.push_back(names);
        }

        for (const auto &out_item: problem->get_stage().get_n_tensor_outputs()) {
            for (unsigned int i = 0; i < out_item.first.second; i++) {
                tensor_values.push_back(problem->get_mesh()->project_n_tensor_qp_field_el_wise(out_item.first.first,
                                                                                               out_item.second,
                                                                                               i));
                auto names = OutputFlags::to_string<dim>(out_item.first.first, i);
                tensor_names.push_back(names);
            }
        }

        vector<Vector<double>> mesh_values;
        vector<string> mesh_names;
        for (const auto &out_item: problem->get_stage().get_mesh_outputs()) {
            mesh_names.push_back(OutputFlags::to_string(out_item));
            mesh_values.push_back(problem->get_mesh()->mesh_output_values(out_item));
        }

        if (problem->get_mesh()->get_this_mpi_processes() == 0) {

            DataOut<dim> data_out;
            data_out.attach_dof_handler(problem->get_mesh()->get_dof_handler());

            vector<DataComponentInterpretation::DataComponentInterpretation>
                    data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
            vector<DataComponentInterpretation::DataComponentInterpretation>
                    tensor_component_interpretation(dim * dim,
                                                    DataComponentInterpretation::component_is_part_of_tensor);

            data_out.add_data_vector(localized_solution, "displacement", DataOut<dim>::type_dof_data,
                                     data_component_interpretation);

            std::vector<unsigned int> partition_int(problem->get_mesh()->get_triangulation().n_active_cells());
            GridTools::get_subdomain_association(problem->get_mesh()->get_triangulation(), partition_int);

            const Vector<double> partitioning(partition_int.begin(),
                                              partition_int.end());

            data_out.add_data_vector(partitioning, "partitioning");

            for (unsigned int i = 0; i < scalar_values.size(); ++i) {
                data_out.add_data_vector(problem->get_mesh()->get_scalar_dof_handler(),
                                         scalar_values.at(i),
                                         scalar_names.at(i));
            }

            for (unsigned int i = 0; i < vector_values.size(); ++i) {
                data_out.add_data_vector(problem->get_mesh()->get_dof_handler(),
                                         vector_values.at(i),
                                         vector_names.at(i),
                                         data_component_interpretation);
            }

            for (unsigned int i = 0; i < tensor_values.size(); ++i) {
                data_out.add_data_vector(problem->get_mesh()->get_tensor_dof_handler(),
                                         tensor_values.at(i),
                                         tensor_names.at(i),
                                         tensor_component_interpretation);
            }

            for (unsigned int i = 0; i < mesh_values.size(); ++i) {
                data_out.add_data_vector(mesh_values.at(i),
                                         mesh_names.at(i));
            }


            data_out.build_patches();
            string rel_vtu_name = problem->get_name()
                                  + "-s-" + to_string(problem->get_time()->get_stage())
                                  + "-t-" + to_string(problem->get_time()->get_timestep()) + ".vtu";

            string name = output_dir_name + rel_vtu_name;
            ofstream output(name);
            data_out.write_vtu(output);
            static vector<pair<double, string>> times_and_names;
            times_and_names.emplace_back(pair<double, string>(problem->get_time()->current(), rel_vtu_name));
            string pvd_name = output_dir_name + problem->get_name() + ".pvd";
            ofstream pvd_output(pvd_name);
            DataOutBase::write_pvd_record(pvd_output, times_and_names);

        }

    }


}

#endif //SOLVER_SOLVER_H
