//
// Created by alhei on 2022/08/29.
//

#ifndef SOLVER_POLYCRYSTALPBC_H
#define SOLVER_POLYCRYSTALPBC_H

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


using namespace std;
using namespace dealii;


template<unsigned int dim>
class PolycrystalPBC {
public:
    explicit PolycrystalPBC(unsigned int n_refinements = 0);

    Triangulation<dim> triangulation;
    map<unsigned int, Material<dim> *> materials;
    vector<Stage<dim>> stages;
    Time time;

protected:

private:

    void read_and_refine_mesh();

    const unsigned int n_refinements;

    array<unsigned int, dim> range;
    const unsigned int order = 1;
    const unsigned int ref = 6;
};

template<unsigned int dim>
PolycrystalPBC<dim>::PolycrystalPBC(unsigned int n_refinements):
        n_refinements(n_refinements) {

    iota(range.begin(), range.end(), 0);
    read_and_refine_mesh();

    string name = "polycrystal-psc-ref-"+ to_string(ref);
    const double kappa = 44.12e3;
    const double mu = 16.92e3;
    const double mult = 1e1;

    const unsigned int n_grains = 8;

    materials[/*material id*/ 0] = new NeoHookean<dim>(/*material id*/ 0, kappa, mu);

    vector<unsigned int> material_ids({0});

    for (unsigned int i = 0; i<n_grains; i++ ) {
        material_ids.push_back(i+1);
        materials[/*material id*/ i+1] = new NeoHookean<dim>(/*material id*/ i+1, mult * kappa, mult * mu);
    }

    FEMesh<dim> mesh(order,
                     materials,
                     triangulation);

    unsigned int n_steps = 50;
    unsigned int n_out = 25;
    double mag = -0.5;

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

    vector<OutputFlags::MeshOutputFlag> mesh_outputs({OutputFlags::MeshOutputFlag::MATERIAL_ID,
                                                      OutputFlags::MeshOutputFlag::BOUNDARY_ID});

    map<OutputFlags::ScalarOutputFlag, vector<unsigned int>> averaged_scalar_outputs;
    averaged_scalar_outputs[OutputFlags::ScalarOutputFlag::ELASTIC_STRAIN_ENERGY] = material_ids;

    map<OutputFlags::TensorOutputFlag, vector<unsigned int>> averaged_tensor_outputs;
    averaged_tensor_outputs[OutputFlags::TensorOutputFlag::F] = material_ids;
    averaged_tensor_outputs[OutputFlags::TensorOutputFlag::STRESS] = material_ids;


    Stage<dim> first_stage(&time,
                           vector<DirichletBoundaryCondition<dim> *>({&pbc}),
                           vector<NeumannBoundaryCondition<dim> *>({}),
                           scalar_outputs,
                           vector_outputs,
                           tensor_outputs,
                           mesh_outputs,
                           averaged_scalar_outputs,
                           {},
                           averaged_tensor_outputs);

    stages.push_back(first_stage);


    Problem<dim> problem(name, &mesh, stages, &time);
    Solver::Solver<dim> solver(&problem);
    solver.solve();

}

template<unsigned int dim>
void PolycrystalPBC<dim>::read_and_refine_mesh() {
    ifstream input_file("../src/examples/polycrystal/polycrystal-" + to_string(ref) + ".dtri");
    {
        boost::archive::text_iarchive ia(input_file);
        triangulation.load(ia, 0);
    }

    if (n_refinements > 0) triangulation.refine_global(n_refinements);
}

#endif //SOLVER_POLYCRYSTALPBC_H
