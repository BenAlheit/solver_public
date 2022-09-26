//
// Created by alhei on 2022/09/01.
//

#ifndef SOLVER_CUBE_H
#define SOLVER_CUBE_H

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>

#include "../materials/OutputFlags.h"
#include "../problem/FEMesh.h"
#include "../Solver.h"

using namespace std;
using namespace dealii;

template<unsigned int dim>
class Cube {
public:
    explicit Cube(unsigned int n_refinements = 1);

    Triangulation<dim> triangulation;
    map<unsigned int, Material<dim> *> materials;
    vector<Stage<dim>> stages;
    Time time;

private:

    void make_mesh();

    const unsigned int n_refinements;

    Point<dim> p1 = Point<dim>({0, 0, 0});
    Point<dim> p2 = Point<dim>({1, 1, 1});
    array<unsigned int, dim> range;
    const unsigned int order = 1;

};

template<unsigned int dim>
Cube<dim>::Cube(unsigned int n_refinements) : n_refinements(n_refinements) {
    iota(range.begin(), range.end(), 0);

    make_mesh();
    string name = "parallel-elastic-cube";
    const double kappa = 44.12e3;
    const double mu = 16.92e3;
    materials[/*material id*/ 0] = new NeoHookean<dim>(/*material id*/ 0, kappa, mu);

    unsigned int n_steps = 50;
    unsigned int n_out = 25;
    double pull = 1.;

    time = Time(/*end_time*/ 1,
                             n_steps,
                             n_out);

    vector<DirichletBoundaryCondition<dim> *> dbcs;
    for (const auto &i: range)
        dbcs.push_back(new Slider<dim>(1 + i, i));

    dbcs.push_back(new DirichletBoundaryCondition<dim>( /*boundary_id*/ 4,
                                                        /*components*/ vector<unsigned int>({0}),
                                                        /*magnitudes*/ vector<double>({pull})));

    vector<unsigned int> material_ids({0});

    map<OutputFlags::ScalarOutputFlag, vector<unsigned int>> scalar_outputs;
    scalar_outputs[OutputFlags::ScalarOutputFlag::J] = material_ids;
    scalar_outputs[OutputFlags::ScalarOutputFlag::P] = material_ids;

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

    Stage<dim> first_stage(&time,
                           dbcs,
                           vector<NeumannBoundaryCondition<dim> *>({}),
                           scalar_outputs,
                           vector_outputs,
                           tensor_outputs,
                           mesh_outputs);

    stages.push_back(first_stage);
    FEMesh<dim> mesh(order,
                     materials,
                     triangulation);

    Problem<dim> problem(name, &mesh, stages, &time);
    Solver::Solver<dim> solver(&problem);
    solver.solve();

}

template<unsigned int dim>
void Cube<dim>::make_mesh() {
//    GridGenerator::subdivided_hyper_rectangle(triangulation, config->repetitions, config->p1, config->p2);
    GridGenerator::hyper_rectangle(triangulation, p1, p2);

    for (const auto &cell: triangulation.cell_iterators())
        cell->set_material_id(0);

    for (auto &face: triangulation.active_face_iterators()) {
        if (face->at_boundary()) {
            for (const auto &i: range) {
                if (almost_equals(face->center()[i], p1[i])) {
                    face->set_boundary_id(1 + i);
                    break;
                } else if (almost_equals(face->center()[i], p2[i])) {
                    face->set_boundary_id(1 + dim + i);
                    break;
                }
            }
        }
    }

    if (n_refinements > 0) triangulation.refine_global(n_refinements);

}

#endif //SOLVER_CUBE_H
