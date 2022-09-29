//
// Created by alhei on 2022/08/29.
//

#ifndef SOLVER_FCCPOLYCRYSTALIMPPBC_H
#define SOLVER_FCCPOLYCRYSTALIMPPBC_H

#include <boost/archive/text_iarchive.hpp>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>

#include "../../materials/OutputFlags.h"
#include "../../problem/FEMesh.h"
#include "../../utils/utils.h"
#include "../../Solver.h"
#include <random>

using namespace std;
using namespace dealii;


template<unsigned int dim>
class FCCPolycrystalIMPPBC {
public:
    explicit FCCPolycrystalIMPPBC(unsigned int n_refinements = 0);

    Triangulation<dim> triangulation;
    map<unsigned int, Material<dim> *> materials;
    vector<Stage<dim>> stages;
    Time time;

protected:

private:

    void read_and_refine_mesh();

    vector<double> random_rotation();

    const unsigned int n_refinements;

    array<unsigned int, dim> range;
    const unsigned int order = 1;
    const unsigned int ref = 6;
    const bool imp = false;
    unsigned int seed = 0;
};

template<unsigned int dim>
FCCPolycrystalIMPPBC<dim>::FCCPolycrystalIMPPBC(unsigned int n_refinements):
        n_refinements(n_refinements) {

    iota(range.begin(), range.end(), 0);
    read_and_refine_mesh();

    unsigned int n_steps = 50;
    unsigned int n_out = 25;
    double mag = -0.12;

    string name = "fcc-polycrystal-psc-ref-"+ to_string(ref);
    if(imp)
        name += "-imp";

    const double kappa = 44.12e3;
    const double mu = 16.92e3;
    const double mult_p = 2;

    const unsigned int n_grains = 8;

    materials[/*material id*/ 1] = new NeoHookean<dim>(/*material id*/ 1, mult_p*kappa, mult_p*mu);

    vector<unsigned int> material_ids({1});
    vector<unsigned int> viscoplastic_material_ids;

    const unsigned int n_sys = 12;
    const double visc = 500;
    const double m = 0.5;
    const double sig_y = 60;
    const double q = 0.5;

    vector<FCCState<dim>*> FCC_example_states;
    vector<vector<double>> ref_euler_angles;
    vector<NeoHookean<dim>*> elastic_bases;

    for (unsigned int i = 0; i<n_grains; i++ ) {
        ref_euler_angles.push_back(random_rotation());
        FCC_example_states.push_back(new FCCState<dim>(ref_euler_angles.at(i)));
        material_ids.push_back(i+2);
        viscoplastic_material_ids.push_back(i+2);
        elastic_bases.push_back(new NeoHookean<dim>(/*material id*/ i+2, kappa, mu));

        materials[/*material id*/ i+2] = new ExplicitRateCrystalDependentPlasticity<dim>(elastic_bases.at(i),
                                                                                         FCC_example_states.at(i),
                                                                                         sig_y,
                                                                                         visc,
                                                                                         m,
                                                                                         q);
    }

    FEMesh<dim> mesh(order,
                     materials,
                     triangulation);



    time = Time(/*end_time*/ 1,
                             n_steps,
                             n_out);

    typedef pair<unsigned int, unsigned int> b_pair;
    vector<b_pair> b_pairs({b_pair(1, 4),
                            b_pair(2, 5),
                            b_pair(3, 6)});

    Tensor<2, dim> Grad_u_end;

    Grad_u_end[1][1] = mag;
    Grad_u_end[0][0] = 1/(1+mag) - 1;

    PeriodicBoundaryCondition<dim> pbc(mesh.get_dof_handler(),
                                       mesh.get_triangulation(),
                                       b_pairs,
                                       Grad_u_end);

    map<OutputFlags::ScalarOutputFlag, vector<unsigned int>> scalar_outputs;
    scalar_outputs[OutputFlags::ScalarOutputFlag::J] = material_ids;
    scalar_outputs[OutputFlags::ScalarOutputFlag::P] = material_ids;
    scalar_outputs[OutputFlags::ScalarOutputFlag::ELASTIC_STRAIN_ENERGY] = material_ids;

    map<OutputFlags::VectorOutputFlag, vector<unsigned int>> vector_outputs;
    vector_outputs[OutputFlags::VectorOutputFlag::NB1] = material_ids;
    vector_outputs[OutputFlags::VectorOutputFlag::NB2] = material_ids;
    vector_outputs[OutputFlags::VectorOutputFlag::NB3] = material_ids;

    map<OutputFlags::TensorOutputFlag, vector<unsigned int>> tensor_outputs;
    tensor_outputs[OutputFlags::TensorOutputFlag::F] = material_ids;
    tensor_outputs[OutputFlags::TensorOutputFlag::STRAIN] = material_ids;
    tensor_outputs[OutputFlags::TensorOutputFlag::STRESS] = material_ids;
    tensor_outputs[OutputFlags::TensorOutputFlag::FP] = viscoplastic_material_ids;
    tensor_outputs[OutputFlags::TensorOutputFlag::FE] = viscoplastic_material_ids;

    vector<OutputFlags::MeshOutputFlag> mesh_outputs({OutputFlags::MeshOutputFlag::MATERIAL_ID,
                                                      OutputFlags::MeshOutputFlag::BOUNDARY_ID});

    map<OutputFlags::ScalarOutputFlag, vector<unsigned int>> averaged_scalar_outputs;
    averaged_scalar_outputs[OutputFlags::ScalarOutputFlag::ELASTIC_STRAIN_ENERGY] = material_ids;

    map<OutputFlags::TensorOutputFlag, vector<unsigned int>> averaged_tensor_outputs;
    averaged_tensor_outputs[OutputFlags::TensorOutputFlag::F] = material_ids;
    averaged_tensor_outputs[OutputFlags::TensorOutputFlag::STRESS] = material_ids;
    averaged_tensor_outputs[OutputFlags::TensorOutputFlag::FirstPiolaStress] = material_ids;


    map<nScalarOutput, vector<unsigned int>, nOutputHash> n_scalar_outputs;
    n_scalar_outputs.insert({OutputFlags::nScalarOutput(OutputFlags::nScalarOutputFlag::H, n_sys), viscoplastic_material_ids});
    n_scalar_outputs.insert({OutputFlags::nScalarOutput(OutputFlags::nScalarOutputFlag::ta, n_sys), viscoplastic_material_ids});
    n_scalar_outputs.insert({OutputFlags::nScalarOutput(OutputFlags::nScalarOutputFlag::nu, n_sys), viscoplastic_material_ids});


    map<nVectorOutput, vector<unsigned int>, nOutputHash> n_vector_outputs;
    n_vector_outputs.insert({OutputFlags::nVectorOutput(OutputFlags::nVectorOutputFlag::M, n_sys), viscoplastic_material_ids});
    n_vector_outputs.insert({OutputFlags::nVectorOutput(OutputFlags::nVectorOutputFlag::S, n_sys), viscoplastic_material_ids});
    n_vector_outputs.insert({OutputFlags::nVectorOutput(OutputFlags::nVectorOutputFlag::CRYSTAL_BASIS, dim), viscoplastic_material_ids});

    map<nTensorOutput, vector<unsigned int>, nOutputHash> n_tensor_outputs;
    n_tensor_outputs.insert({OutputFlags::nTensorOutput(OutputFlags::nTensorOutputFlag::SYS, n_sys), viscoplastic_material_ids});


    Stage<dim> first_stage(&time,
                           vector<DirichletBoundaryCondition<dim> *>({&pbc}),
                           vector<NeumannBoundaryCondition<dim> *>({}),
                           scalar_outputs,
                           vector_outputs,
                           tensor_outputs,
                           mesh_outputs,
                           averaged_scalar_outputs,
                           {},
                           averaged_tensor_outputs,
                           time.end(),
                           time.get_n_steps(),
                           time.get_n_steps_out(),
                           n_scalar_outputs,
                           n_vector_outputs,
                           n_tensor_outputs);

    stages.push_back(first_stage);


    Problem<dim> problem(name, &mesh, stages, &time);
    Solver::Solver<dim> solver(&problem);
    solver.solve();

}

template<unsigned int dim>
void FCCPolycrystalIMPPBC<dim>::read_and_refine_mesh() {
    string file_name = "../src/examples/fcc-polycrystal/polycrystal-";
    if(imp)
        file_name += "imp-";
    file_name += "ngb-"+ to_string(ref) + ".dtri";

    ifstream input_file(file_name.c_str());
    {
        boost::archive::text_iarchive ia(input_file);
        triangulation.load(ia, 0);
    }

    if (n_refinements > 0) triangulation.refine_global(n_refinements);
}

template<unsigned int dim>
vector<double> FCCPolycrystalIMPPBC<dim>::random_rotation() {

    seed++;
    mt19937 gen(seed);
    uniform_int_distribution<> distr(0, 360);

    return vector<double>({distr(gen), distr(gen), distr(gen)});
}

#endif //SOLVER_FCCPOLYCRYSTALIMPPBC_H
